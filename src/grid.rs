use crate::*;

#[derive(KernelArg)]
pub struct CellParticleList {
    pub head: Buffer<u32>,
    pub next: Buffer<u32>,
    #[luisa(exclude)]
    pub reset: Kernel<(Buffer<u32>,)>,
}
impl CellParticleList {
    pub fn new(device: Device, ncell: usize, nparticles: usize) -> Self {
        let head = device.create_buffer(ncell).unwrap();
        let next = device.create_buffer(nparticles).unwrap();
        let reset = device
            .create_kernel::<(Buffer<u32>,)>(&|a| {
                let tid = dispatch_id().x();
                a.write(tid, u32::MAX);
            })
            .unwrap();
        Self { head, reset, next }
    }
    pub fn reset(&self) {
        self.reset
            .dispatch([self.head.len() as u32, 1, 1], &self.head)
            .unwrap();
        self.reset
            .dispatch([self.next.len() as u32, 1, 1], &self.next)
            .unwrap();
    }
    pub fn append(&self, cell_index: Expr<u32>, particle_index: Expr<u32>) {
        let head = self.head.var();
        while_!(const_(true), {
            let old = head.read(cell_index);
            let cur = head.atomic_compare_exchange(cell_index, old, particle_index);
            if_!(cur.cmpeq(old), {
                self.next.var().write(particle_index, old);
                break_();
            })
        });
    }
    pub fn for_each_particle_in_cell(&self, cell: Expr<u32>, f: impl FnOnce(Expr<u32>)) {
        let head = self.head.var();
        let next = self.next.var();
        let p = var!(u32, head.read(cell));
        while_!(p.load().cmplt(u32::MAX), {
            f(p.load());
            p.store(next.read(p.load()));
        });
    }
}
#[derive(KernelArg)]
pub struct Grid<T: Value> {
    pub values: Buffer<T>,
    pub dx: f32,
    #[luisa(exclude)]
    pub dimension: usize,
    #[luisa(exclude)]
    pub res: [u32; 3],
    #[luisa(exclude)]
    pub origin: [f32; 3],
    #[luisa(exclude)]
    pub cell_particle_list: Option<CellParticleList>,
    #[luisa(exclude)]
    device: Device,
}

impl<T: Value> Grid<T> {
    pub fn new(device: Device, res: [u32; 3], dimension: usize, origin: [f32; 3], dx: f32) -> Self {
        if dimension == 2 {
            assert_eq!(res[2], 1);
        }
        let values = device
            .create_buffer::<T>((res[0] * res[1] * res[2]) as usize)
            .unwrap();
        Self {
            dimension,
            values,
            res,
            origin,
            cell_particle_list: None,
            device,
            dx,
        }
    }
    pub fn init_particle_list(&mut self, count: usize) {
        self.cell_particle_list = Some(CellParticleList::new(
            self.device.clone(),
            self.values.len(),
            count,
        ));
    }
    pub fn reset_particle_list(&self) {
        if let Some(list) = &self.cell_particle_list {
            list.reset();
        }
    }
    pub fn linear_index_host(&self, p: Uint3) -> u32 {
        if self.dimension == 2 {
            p.x + p.y * self.res[0]
        } else {
            p.x + p.y * self.res[0] + p.z * self.res[0] * self.res[1]
        }
    }
    pub fn linear_index(&self, p: Expr<Uint3>) -> Expr<u32> {
        assert(!self.oob(p.int()));
        let p = p.clamp(
            make_uint3(0, 0, 0),
            make_uint3(self.res[0] - 1, self.res[1] - 1, self.res[2] - 1),
        );
        if self.dimension == 2 {
            p.x() + p.y() * const_(self.res[0])
        } else {
            p.x() + p.y() * const_(self.res[0]) + p.z() * const_(self.res[0] * self.res[1])
        }
    }
    pub fn oob(&self, p: Expr<Int3>) -> Bool {
        if self.dimension == 2 {
            let oob = p.xy().cmplt(Int2Expr::zero())
                | p.xy().cmpge(make_uint2(self.res[0], self.res[1]).int());
            oob.any()
        } else {
            let oob = p.cmpge(make_uint3(self.res[0], self.res[1], self.res[2]).int())
                | p.cmplt(Int3Expr::zero());
            oob.any()
        }
    }
    pub fn at_index_clamped(&self, p: Expr<Uint3>) -> Expr<T> {
        if self.dimension == 2 {
            let p = p.min(make_uint3(self.res[0] - 1, self.res[1] - 1, 0));
            self.at_index(p)
        } else {
            let p = p.min(make_uint3(
                self.res[0] - 1,
                self.res[1] - 1,
                self.res[2] - 1,
            ));
            self.at_index(p)
        }
    }
    pub fn clamp(&self, p: Expr<Int3>) -> Expr<Uint3> {
        assert(p.cmpge(0).all());
        let p = p.uint();
        if self.dimension == 2 {
            let p = p.min(make_uint3(self.res[0] - 1, self.res[1] - 1, 0));
            p
        } else {
            let p = p.min(make_uint3(
                self.res[0] - 1,
                self.res[1] - 1,
                self.res[2] - 1,
            ));
            p
        }
    }
    pub fn at_index(&self, p: Expr<Uint3>) -> Expr<T> {
        // if cfg!(debug_assertions) {
        //     if_!(self.oob(p.int()), {
        //         cpu_dbg!(p.int());
        //         cpu_dbg!(make_uint3(self.res[0], self.res[1], self.res[2]));
        //     });
        // }
        assert(!self.oob(p.int()));
        let index = self.linear_index(p);
        self.values.var().read(index)
    }
    pub fn set_index(&self, p: Expr<Uint3>, v: Expr<T>) {
        assert(!self.oob(p.int()));
        let index = self.linear_index(p);
        self.values.var().write(index, v);
    }
    pub fn pos_f_to_i(&self, p: Expr<Float3>) -> Expr<Int3> {
        let p = (p - make_float3(self.origin[0], self.origin[1], self.origin[2])) / self.dx;
        p.int()
    }
    pub fn pos_i_to_f(&self, p: Expr<Int3>) -> Expr<Float3> {
        p.float() * self.dx + make_float3(self.origin[0], self.origin[1], self.origin[2])
    }
    pub fn cell_center(&self, p: Expr<Int3>) -> Expr<Float3> {
        self.pos_i_to_f(p) + self.dx * 0.5
    }
    pub fn get_cell(&self, p: Expr<Float3>) -> Expr<Int3> {
        self.pos_f_to_i(p + self.dx * 0.5)
    }
    pub fn add_to_cell(&self, p: Expr<Float3>, i: Expr<u32>) {
        let ip = self.pos_f_to_i(p + self.dx * 0.5);
        // let ip = self.clamp(ip);
        // let linear_index = self.linear_index(ip.uint());
        //     self.cell_particle_list
        //         .as_ref()
        //         .unwrap()
        //         .append(linear_index, i);
        let oob = self.oob(ip);
        // cpu_dbg!(ip);
        // cpu_dbg!(p);
        // cpu_dbg!(const_(self.dx));
        // cpu_dbg!((self.pos_i_to_f(ip) - p).length());
        if_!(!oob, {
            let linear_index = self.linear_index(ip.uint());
            self.cell_particle_list
                .as_ref()
                .unwrap()
                .append(linear_index, i);
        }, else{
            cpu_dbg!(ip);
            cpu_dbg!(make_uint3(self.res[0], self.res[1], self.res[2]));
        });
    }
    pub fn for_each_neighbor_node(&self, p: Expr<Float3>, f: impl Fn(Expr<Uint3>)) {
        let ip = self.pos_f_to_i(p);
        let map = |offset: [i32; 3]| {
            let offset = make_int3(offset[0], offset[1], offset[2]);
            let neighbor = ip + offset;
            if_!(!self.oob(neighbor), { f(neighbor.uint()) })
        };
        map([0, 0, 0]);
        map([1, 0, 0]);
        map([0, 1, 0]);
        map([1, 1, 0]);
        if self.dimension == 3 {
            map([0, 0, 1]);
            map([1, 0, 1]);
            map([0, 1, 1]);
            map([1, 1, 1]);
        }
    }
    pub fn for_each_particle_in_cell(&self, cell: Expr<Uint3>, f: impl Fn(Expr<u32>)) {
        let index = self.linear_index(cell);
        self.cell_particle_list
            .as_ref()
            .unwrap()
            .for_each_particle_in_cell(index, f);
    }
    pub fn for_each_particle_in_neighbor(
        &self,
        node: Expr<Uint3>,
        lo: [i32; 3],
        hi: [i32; 3],
        f: impl Fn(Expr<u32>),
    ) {
        let f = &f;
        let map = |offset: [i32; 3]| {
            let offset = make_int3(offset[0], offset[1], offset[2]);
            let neighbor = node.int() + offset;
            if_!(!self.oob(neighbor), {
                let neighbor = neighbor.uint();
                self.for_each_particle_in_cell(neighbor, f)
            })
        };
        // map([0, 0, 0]);
        // map([-1, 0, 0]);
        // map([0, -1, 0]);
        // map([-1, -1, 0]);
        // if self.dimension == 3 {
        //     map([0, 0, -1]);
        //     map([-1, 0, -1]);
        //     map([0, -1, -1]);
        //     map([-1, -1, -1]);
        // }
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                if self.dimension == 3 {
                    for k in lo[2]..=hi[2] {
                        map([i, j, k]);
                    }
                } else {
                    map([i, j, 0]);
                }
            }
        }
    }
}

impl Grid<f32> {
    pub fn interpolate(&self, p: Expr<Float3>) -> Expr<f32> {
        if self.dimension == 2 {
            let p = (p - make_float3(self.origin[0], self.origin[1], 0.0)) / self.dx;
            let ip = p.floor().int();

            let offset = p - ip.float();

            let v00 = self.at_index_clamped(ip.uint());
            let v01 = self.at_index_clamped(ip.uint() + make_uint3(1, 0, 0));
            let v10 = self.at_index_clamped(ip.uint() + make_uint3(0, 1, 0));
            let v11 = self.at_index_clamped(ip.uint() + make_uint3(1, 1, 0));
            let v0 = (1.0 - offset.x()) * v00 + offset.x() * v01;
            let v1 = (1.0 - offset.x()) * v10 + offset.x() * v11;
            let v = (1.0 - offset.y()) * v0 + offset.y() * v1;
            v
        } else {
            let p = (p - make_float3(self.origin[0], self.origin[1], self.origin[2])) / self.dx;
            let ip = p.floor().int();
            let offset = p - ip.float();
            let v000 = self.at_index_clamped(ip.uint());
            let v001 = self.at_index_clamped(ip.uint() + make_uint3(1, 0, 0));
            let v010 = self.at_index_clamped(ip.uint() + make_uint3(0, 1, 0));
            let v011 = self.at_index_clamped(ip.uint() + make_uint3(1, 1, 0));

            let v00 = (1.0 - offset.x()) * v000 + offset.x() * v001;
            let v01 = (1.0 - offset.x()) * v010 + offset.x() * v011;
            let v0 = (1.0 - offset.y()) * v00 + offset.y() * v01;

            let v100 = self.at_index_clamped(ip.uint());
            let v101 = self.at_index_clamped(ip.uint() + make_uint3(1, 0, 1));
            let v110 = self.at_index_clamped(ip.uint() + make_uint3(0, 1, 1));
            let v111 = self.at_index_clamped(ip.uint() + make_uint3(1, 1, 1));

            let v10 = (1.0 - offset.x()) * v100 + offset.x() * v101;
            let v11 = (1.0 - offset.x()) * v110 + offset.x() * v111;
            let v1 = (1.0 - offset.y()) * v10 + offset.y() * v11;

            let v = (1.0 - offset.z()) * v0 + offset.z() * v1;
            v
        }
    }
    pub fn at_index_or_zero(&self, p: Expr<Int3>) -> Expr<f32> {
        let oob = self.oob(p);
        if self.dimension == 2 {
            if_!(oob, {
                const_(0.0f32)
            }, else{
                let index = self.linear_index(p.uint());
                self.values.var().read(index)
            })
        } else {
            if_!(oob, {
                const_(0.0f32)
            }, else{
                let index = self.linear_index(p.uint());
                self.values.var().read(index)
            })
        }
    }
}

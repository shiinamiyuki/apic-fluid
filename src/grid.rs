use crate::*;

#[derive(KernelArg)]
pub struct CellParticleList {
    pub head: Buffer<u32>,
    pub next: Buffer<u32>,
    #[luisa(exclude)]
    pub reset: Kernel<(Buffer<u32>,)>,
}
impl CellParticleList {
    pub fn new(device: Device, nparticles: usize) -> Self {
        let head = device.create_buffer(nparticles).unwrap();
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
    #[luisa(exclude)]
    pub dimension: usize,
    #[luisa(exclude)]
    pub res: [u32; 3],
    #[luisa(exclude)]
    pub shift: [T; 3],
    #[luisa(exclude)]
    pub cell_particle_list: Option<CellParticleList>,
    #[luisa(exclude)]
    device: Device,
}

impl<T: Value> Grid<T> {
    pub fn new(device: Device, res: [u32; 3], dimension: usize, shift: [T; 3]) -> Self {
        if dimension == 2 {
            assert_eq!(res[2], 1);
        }
        let values = device
            .create_buffer::<T>((res[0] * res[1] * res[2]) as usize * dimension)
            .unwrap();
        Self {
            dimension,
            values,
            res,
            shift,
            cell_particle_list: None,
            device,
        }
    }
    pub fn init_particle_list(&mut self, count: usize) {
        self.cell_particle_list = Some(CellParticleList::new(self.device.clone(), count));
    }
    pub fn linear_index(&self, p: Expr<UVec3>) -> Expr<u32> {
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
    pub fn oob(&self, p: Expr<IVec3>) -> Bool {
        if self.dimension == 2 {
            let oob = p.xy().cmplt(IVec2Expr::zero())
                | p.xy().cmpge(make_uint2(self.res[0], self.res[1]).int());
            oob.any()
        } else {
            let oob = p.cmpge(make_uint3(self.res[0], self.res[1], self.res[2]).int())
                | p.cmpge(IVec3Expr::zero());
            oob.any()
        }
    }
    pub fn at_index(&self, p: Expr<UVec3>) -> Expr<T> {
        assert(!self.oob(p.int()));
        let index = self.linear_index(p);
        self.values.var().read(index)
    }
    pub fn set_index(&self, p: Expr<UVec3>, v: Expr<T>) {
        assert(!self.oob(p.int()));
        let index = self.linear_index(p);
        self.values.var().write(index, v);
    }
    pub fn add_to_cell(&self, p: Expr<Vec3>, i: Expr<u32>) {
        let oob = self.oob(p.int());
        let linear_index = self.linear_index(p.uint());
        if_!(!oob, {
            self.cell_particle_list
                .as_ref()
                .unwrap()
                .append(linear_index, i);
        });
    }
    pub fn for_each_particle_in_cell(&self, cell: Expr<UVec3>, f: impl FnOnce(Expr<u32>)) {
        let index = self.linear_index(cell);
        self.cell_particle_list
            .as_ref()
            .unwrap()
            .for_each_particle_in_cell(index, f);
    }
}

impl Grid<f32> {
    pub fn bilinear(&self, p: Expr<Vec3>) -> Expr<f32> {
        if self.dimension == 2 {
            let p = p - make_float3(self.shift[0], self.shift[1], 0.0);
            let ip = p.floor().int();
            let offset = p - ip.float();
            let v00 = self.at_index(ip.uint());
            let v01 = self.at_index(ip.uint() + make_uint3(1, 0, 0));
            let v10 = self.at_index(ip.uint() + make_uint3(0, 1, 0));
            let v11 = self.at_index(ip.uint() + make_uint3(1, 1, 0));
            let v0 = (1.0 - offset.x()) * v00 + offset.x() * v01;
            let v1 = (1.0 - offset.x()) * v10 + offset.x() * v11;
            let v = (1.0 - offset.y()) * v0 + offset.y() * v1;
            v
        } else {
            todo!()
        }
    }
    pub fn at_index_or_zero(&self, p: Expr<IVec3>) -> Expr<f32> {
        let oob = self.oob(p);
        if self.dimension == 2 {
            if_!(oob, {
                const_(0.0f32)
            }, else{
                let index = self.linear_index(p.as_uvec3());
            self.values.var().read(index)
            })
        } else {
            if_!(oob, {
                const_(0.0f32)
            }, else{
                let index = self.linear_index(p.as_uvec3());
            self.values.var().read(index)
            })
        }
    }
}

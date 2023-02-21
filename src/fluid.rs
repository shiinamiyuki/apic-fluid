use crate::{
    sparse::{solver::PcgSolver, SparseMatrix},
    *,
};
#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
}

pub struct Particles {
    pub head: Grid<u32>,
    pub dimension: usize,
    pub next: Buffer<u32>,
    pub particles: Buffer<Particle>,
    pub h: f32,
    pub res: [u32; 3],
    pub reset: Kernel<(Buffer<u32>,)>,
}
impl Particles {
    pub fn new(
        device: Device,
        res: [u32; 3],
        dimension: usize,
        h: f32,
        particles: Vec<Particle>,
    ) -> Self {
        let head = Grid::<u32>::new(device.clone(), res, dimension);
        head.values.view(..).fill(u32::MAX);
        let next = device.create_buffer(particles.len()).unwrap();
        let reset = device
            .create_kernel::<(Buffer<u32>,)>(&|a| {
                let tid = dispatch_id().x();
                a.write(tid, u32::MAX);
            })
            .unwrap();
        let particles = device.create_buffer_from_slice(&particles).unwrap();
        Self {
            head,
            reset,
            dimension,
            particles,
            next,
            h,
            res,
        }
    }
    pub fn reset(&self) {
        self.reset
            .dispatch([self.head.values.len() as u32, 1, 1], &self.head.values)
            .unwrap();
        self.reset
            .dispatch([self.particles.len() as u32, 1, 1], &self.head.values)
            .unwrap();
    }
    pub fn get_cell(&self, p: Expr<Vec3>) -> (Bool, Expr<IVec3>) {
        let cell = (p / self.h).int();
        let res = make_uint3(self.res[0], self.res[1], self.res[2]);
        let oob = cell.cmplt(IVec3Expr::zero()).any() | cell.cmpge(res.int()).any();
        (oob, cell)
    }
    pub fn append(&self, p: Expr<Vec3>, i: Expr<u32>) {
        let (oob, cell) = self.get_cell(p);
        if_!(!oob, {
            let cell = cell.uint();
            let cell_index = self.head.linear_index(cell);
            let head = self.head.values.var();
            while_!(const_(true), {
                let old = head.read(cell_index);
                let cur = head.atomic_compare_exchange(cell_index, old, i);
                if_!(cur.cmpeq(old), {
                    self.next.var().write(i, old);
                    break_();
                })
            });
        });
    }
    pub fn for_each_particle_in_cell(&self, cell: Expr<u32>, f: impl FnOnce(Expr<u32>)) {
        let head = self.head.values.var();
        let next = self.next.var();
        let p = var!(u32, head.read(cell));
        while_!(p.load().cmplt(u32::MAX), {
            f(p.load());
            p.store(next.read(p.load()));
        });
    }
}
pub struct Simulation {
    pub u: Grid<f32>,
    pub v: Grid<f32>,
    pub w: Grid<f32>,
    pub p: Grid<f32>,
    pub dimension: usize,
    pub res: [u32; 3],
    pub solver: PcgSolver,
    pub A: Option<SparseMatrix>,
    pub particles_vec: Vec<Particle>,
    pub particles:Option<Particles>
}
impl Simulation {
    pub fn new(device: Device, res: [u32; 3], dimension: usize) -> Self {
        let u = Grid::new(device.clone(), res, dimension);
        let v = Grid::new(device.clone(), res, dimension);
        let w = Grid::new(device.clone(), res, dimension);
        let p = Grid::new(device.clone(), res, dimension);
        let cell_particle_list = Grid::<u32>::new(device.clone(), res, dimension);
        cell_particle_list.values.view(..).fill(u32::MAX);
        Self {
            u,
            v,
            w,
            p,
            dimension,
            res,
            solver: PcgSolver::new(device.clone(), (res[0] * res[1] * res[2]) as usize),
            A: None,
            particles_vec: Vec::new(),
            particles: None,
        }
    }
}

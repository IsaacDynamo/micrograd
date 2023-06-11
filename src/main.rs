use std::{
    cell::RefCell,
    ops::{Add, Deref, Div, Mul, Sub},
};

use rand::distributions::{Distribution, Uniform};

#[derive(Default, Debug)]
struct Pool {
    entries: RefCell<Vec<ValueEntry>>,
    grads: RefCell<Vec<f32>>,
}

impl Pool {
    fn new() -> Self {
        Self::default()
    }

    fn add(&self, entry: ValueEntry) -> Value {
        let mut entries = self.entries.borrow_mut();
        let index = entries.len();
        entries.push(entry);
        Value { pool: self, index }
    }

    fn value(&self, value: f32) -> Value {
        self.add(ValueEntry {
            value,
            children: Vec::new(),
            operation: Operation::Constant,
        })
    }

    fn schedule(&self, index: usize) -> Vec<usize> {
        let entries = self.entries.borrow();
        let mut schedule = Vec::with_capacity(entries.len());
        let mut visited = vec![false; entries.len()];

        fn build_topo(
            entries: &Vec<ValueEntry>,
            schedule: &mut Vec<usize>,
            visited: &mut Vec<bool>,
            index: usize,
        ) {
            if !visited[index] {
                visited[index] = true;
                for &child in entries[index].children.iter() {
                    build_topo(entries, schedule, visited, child)
                }
                schedule.push(index);
            }
        }

        build_topo(entries.deref(), &mut schedule, &mut visited, index);

        schedule
    }

    fn backward(&self, index: usize) {
        let entries = self.entries.borrow();
        let mut grads = self.grads.borrow_mut();

        let local_grad = grads[index];

        let entry = &entries[index];
        match entry.operation {
            Operation::Constant => (),
            Operation::Add => {
                for index in entry.children.iter() {
                    grads[*index] += local_grad;
                }
            }
            Operation::Mul => {
                let ia = entry.children[0];
                let ib = entry.children[1];
                let a = &entries[ia];
                let b = &entries[ib];

                grads[ia] += local_grad * b.value;
                grads[ib] += local_grad * a.value;
            }
            Operation::Tanh => {
                let i = entry.children[0];
                grads[i] += local_grad * (1.0 - entry.value * entry.value);
            }
            Operation::Exp => {
                let i = entry.children[0];
                grads[i] += local_grad * entry.value;
            }
            Operation::Pow(x) => {
                let i = entry.children[0];
                grads[i] += local_grad * x * entries[i].value.powf(x - 1.0);
            }
        }
    }
}

#[derive(Debug)]
enum Operation {
    Constant,
    Add,
    Mul,
    Tanh,
    Exp,
    Pow(f32),
}

#[derive(Debug)]
struct ValueEntry {
    value: f32,
    children: Vec<usize>,
    operation: Operation,
}

#[derive(Clone, Copy, Debug)]
struct Value<'a> {
    pool: &'a Pool,
    index: usize,
}

impl<'a> Value<'a> {
    fn value(&self) -> f32 {
        self.pool.entries.borrow()[self.index].value
    }

    fn value_inc(&self, x: f32) {
        self.pool.entries.borrow_mut()[self.index].value += x;
    }

    fn grad(&self) -> f32 {
        self.pool.grads.borrow()[self.index]
    }

    fn tanh(&self) -> Self {
        self.pool.add(ValueEntry {
            value: self.value().tanh(),
            children: vec![self.index],
            operation: Operation::Tanh,
        })
    }

    fn exp(&self) -> Self {
        self.pool.add(ValueEntry {
            value: self.value().exp(),
            children: vec![self.index],
            operation: Operation::Exp,
        })
    }

    fn pow(&self, x: f32) -> Self {
        self.pool.add(ValueEntry {
            value: self.value().powf(x),
            children: vec![self.index],
            operation: Operation::Pow(x),
        })
    }

    fn backward(&self) {
        let schedule = self.pool.schedule(self.index);

        {
            let entries = self.pool.entries.borrow();
            let mut grads = self.pool.grads.borrow_mut();
            *grads = vec![0.0; entries.len()];
            grads[self.index] = 1.0;
        }

        for &i in schedule.iter().rev() {
            self.pool.backward(i)
        }
    }
}

impl<'a> Mul for Value<'a> {
    type Output = Value<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.pool.add(ValueEntry {
            value: self.value() * rhs.value(),
            children: vec![self.index, rhs.index],
            operation: Operation::Mul,
        })
    }
}

impl<'a> Add for Value<'a> {
    type Output = Value<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        self.pool.add(ValueEntry {
            value: self.value() + rhs.value(),
            children: vec![self.index, rhs.index],
            operation: Operation::Add,
        })
    }
}

impl<'a> Sub for Value<'a> {
    type Output = Value<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (rhs * self.pool.value(-1.0))
    }
}

impl<'a> Div for Value<'a> {
    type Output = Value<'a>;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1.0)
    }
}

struct Neuron<'a> {
    ws: Vec<Value<'a>>,
    b: Value<'a>,
}

impl<'a> Neuron<'a> {
    fn new(pool: &'a Pool, nin: usize) -> Self {
        let dist = Uniform::<f32>::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        Self {
            ws: Vec::from_iter((0..nin).map(|_| pool.value(dist.sample(&mut rng)))),
            b: pool.value(dist.sample(&mut rng)),
        }
    }

    fn call(&self, xs: &[Value<'a>]) -> Value<'a> {
        xs.iter()
            .zip(self.ws.iter())
            .map(|(&x, &w)| x * w)
            .fold(self.b, |a, b| a + b)
            .tanh()
    }

    fn params(&self) -> Vec<Value<'a>> {
        let mut r = self.ws.clone();
        r.push(self.b);
        r
    }
}

struct Layer<'a> {
    neurons: Vec<Neuron<'a>>,
}

impl<'a> Layer<'a> {
    fn new(pool: &'a Pool, nin: usize, nout: usize) -> Self {
        Self {
            neurons: Vec::from_iter((0..nout).map(|_| Neuron::new(pool, nin))),
        }
    }

    fn call(&self, xs: &[Value<'a>]) -> Vec<Value<'a>> {
        self.neurons.iter().map(|n| n.call(xs)).collect()
    }

    fn params(&self) -> Vec<Value<'a>> {
        self.neurons.iter().flat_map(|n| n.params()).collect()
    }
}

struct Mlp<'a> {
    layers: Vec<Layer<'a>>,
}

impl<'a> Mlp<'a> {
    fn new(pool: &'a Pool, nin: usize, nouts: &[usize]) -> Self {
        let mut shape = vec![nin];
        shape.extend_from_slice(nouts);

        let layers = shape
            .windows(2)
            .map(|s| Layer::new(pool, s[0], s[1]))
            .collect();

        Self { layers }
    }

    fn call(&self, x: &[Value<'a>]) -> Vec<Value<'a>> {
        let mut v = Vec::new();
        let mut q = x;

        for layer in self.layers.iter() {
            v = layer.call(q);
            q = v.as_slice();
        }

        v
    }

    fn params(&self) -> Vec<Value<'a>> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }
}

fn main() {
    let pool = Pool::new();

    let xs = &[
        &[2.0, 3.0, -1.0],
        &[3.0, -1.0, 0.5],
        &[0.5, 1.0, 1.0],
        &[1.0, 1.0, -1.0],
    ];

    let ys = &[1.0, -1.0, -1.0, 1.0];

    let xs = xs
        .iter()
        .map(|&x| x.iter().map(|&v| pool.value(v)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let ys = ys.iter().map(|&y| pool.value(y)).collect::<Vec<_>>();

    let mpl = Mlp::new(&pool, 3, &[4, 4, 1]);

    let params = mpl.params();

    let n = 50;
    for i in 0..n {
        let ypred = xs
            .iter()
            .map(|x| mpl.call(x.as_slice())[0])
            .collect::<Vec<_>>();

        let loss: Value = ypred
            .iter()
            .zip(ys.iter())
            .map(|(&p, &y)| (p - y).pow(2.0))
            .fold(pool.value(0.0), std::ops::Add::add);

        println!("{} {}", i, loss.value());

        loss.backward();

        for p in params.iter() {
            p.value_inc(-0.1 * p.grad());
        }

        if i == (n - 1) {
            println!("{:?}", ypred.iter().map(|v| v.value()).collect::<Vec<_>>());
        }
    }

    println!("{}", pool.entries.borrow().len());
}

#[cfg(test)]
mod test {
    use crate::*;

    macro_rules! assert_approx {
        ($a:expr , $b:expr) => {
            assert!(($a - $b).abs() < 1e-6, "{} !~= {}", $a, $b);
        };
    }

    #[test]
    fn abc() {
        let pool = Pool::new();
        let a = pool.value(2.0f32);
        let b = pool.value(-3.0f32);
        let c = pool.value(10.0f32);
        let d = a * b + c;
        d.backward();

        assert_approx!(a.grad(), -3.0);
        assert_approx!(b.grad(), 2.0);
        assert_approx!(c.grad(), 1.0);
    }

    #[test]
    fn simple() {
        let pool = Pool::new();
        let x1 = pool.value(2.0);
        let x2 = pool.value(0.0);
        let w1 = pool.value(-3.0);
        let w2 = pool.value(1.0);
        let b = pool.value(6.88137358);

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;
        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;
        let o = n.tanh();
        o.backward();

        assert_approx!(w1.grad(), 1.0);
        assert_approx!(w2.grad(), 0.0);
    }

    #[test]
    fn grad_inc() {
        let pool = Pool::new();
        let a = pool.value(4.0);
        let q = a + a;
        q.backward();

        assert_approx!(a.grad(), 2.0);
    }
    #[test]
    fn grad_inc2() {
        let pool = Pool::new();
        let a = pool.value(-2.0);
        let b = pool.value(3.0);
        let f = (a * b) * (a + b);
        f.backward();

        assert_approx!(a.grad(), -3.0);
        assert_approx!(b.grad(), -8.0);
    }
}

#[derive(Debug)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU
}

impl Activation {
    pub fn calculate(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::ReLU => x.max(0.0),
        }
    }

    pub fn calculate_derivative(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => self.calculate(x) * (1.0 - self.calculate(x)),
            Activation::Tanh => 1.0 - self.calculate(x).powi(2),
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
        }
    }
}
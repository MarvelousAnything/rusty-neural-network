#[derive(Debug)]
pub enum Cost {
    MeanSquaredError,
    CrossEntropy,
    // Exponential,
    // HellingerDistance,
    // KullbackLeiblerDivergence,
    // GeneralizedKullbackLeiblerDivergence,
    // ItakuraSaitoDivergence,
}

impl Cost {
    pub fn calculate(&self, targets: &[f32], outputs: &[f32]) -> f32 {
        assert_eq!(targets.len(), outputs.len());
        match self {
            // 1/2 * sum((target - output)^2)
            Cost::MeanSquaredError => {
                let mut cost = 0.0;
                for i in 0..targets.len() {
                    cost += (targets[i] - outputs[i]).powi(2);
                }
                cost / 2.0
            }
            // -sum(target * ln(output) + (1 - target) * ln(1 - output))
            Cost::CrossEntropy => {
                let mut cost = 0.0;
                for i in 0..targets.len() {
                    cost += targets[i] * outputs[i].ln() + (1.0 - targets[i]) * (1.0 - outputs[i]).ln();
                }
                -cost
            }
        }
    }

    pub fn calculate_single(&self, target: f32, output: f32) -> f32 {
        match self {
            // 1/2 * (target - output)^2
            Cost::MeanSquaredError => {
                (target - output).powi(2) / 2.0
            }
            // -target * ln(output) - (1 - target) * ln(1 - output)
            Cost::CrossEntropy => {
                -target * output.ln() - (1.0 - target) * (1.0 - output).ln()
            }
        }
    }

    pub fn calculate_derivative(&self, target: f32, output: f32) -> f32 {
        match self {
            Cost::MeanSquaredError => output - target,
            Cost::CrossEntropy => (output - target)/(output * (1.0 - output)),
        }
    }
}
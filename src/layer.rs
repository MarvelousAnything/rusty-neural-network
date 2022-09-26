use rand::Rng;
use crate::activation::Activation;
use crate::cost::Cost;
use crate::trainer::LayerData;

#[derive(Debug)]
pub enum LayerError {
    SizeMismatch(i32, i32),
}

#[derive(Debug)]
pub struct Layer {
    pub nodes_in: u32,
    pub nodes_out: u32,
    pub weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    cost_gradient_weights: Vec<Vec<f32>>,
    cost_gradient_biases: Vec<f32>,
    pub activation: Activation,
    pub cost: Cost
}

impl Layer {
    pub fn new(nodes_in: u32, nodes_out: u32) -> Layer {
        let mut rng = rand::thread_rng();
        let weights: Vec<Vec<f32>> = (0..nodes_out)
            .map(|_| {
                (0..nodes_in)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect::<Vec<f32>>()
            })
            .collect();
        let biases: Vec<f32> = (0..nodes_out)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect::<Vec<f32>>();
        Layer {
            nodes_in,
            nodes_out,
            weights,
            biases,
            cost_gradient_weights: vec![vec![0.0; nodes_in as usize]; nodes_out as usize],
            cost_gradient_biases: vec![0.0; nodes_out as usize],
            activation: Activation::Sigmoid,
            cost: Cost::MeanSquaredError
        }
    }
    pub fn calculate_outputs(&self, inputs: &[f32]) -> Result<Vec<f32>, LayerError> {
        if inputs.len() != self.nodes_in as usize {
            return Err(LayerError::SizeMismatch(inputs.len() as i32, self.nodes_in as i32));
        }
        let mut outputs = Vec::new();
        for i in 0..self.nodes_out {
            let mut sum = 0.0;
            for j in 0..self.nodes_in {
                sum += inputs[j as usize] * self.weights[i as usize][j as usize];
            }
            sum += self.biases[i as usize];
            outputs.push(self.activation.calculate(sum));
        }
        Ok(outputs)
    }

    pub fn calculate_training_outputs(&self, inputs: &[f32], mut data: LayerData) -> Vec<f32> {
        data.inputs = inputs.to_vec();
        for i in 0..self.nodes_in {
            let mut weighted_input = self.biases[i as usize];
            for j in 0..self.nodes_in {
                weighted_input += inputs[j as usize] * self.weights[j as usize][i as usize];
            }
            data.weighted_inputs[i as usize] = weighted_input;
        }

        for i in 0..data.activations.len() {
            data.activations[i] = self.activation.calculate(data.weighted_inputs[i]);
        }
        data.activations
    }

    pub fn calculate_node_cost(&self, target: f32, output: f32) -> f32 {
        self.cost.calculate_single(target, output)
    }

    pub fn apply_gradients(&mut self, learning_rate: f32) {
        for i in 0..self.nodes_out {
            for j in 0..self.nodes_in {
                self.weights[i as usize][j as usize] += learning_rate * self.cost_gradient_weights[i as usize][j as usize];
            }
            self.biases[i as usize] += learning_rate * self.cost_gradient_biases[i as usize];
        }
    }

    pub fn calculate_output_layer_node_values(&self, data: LayerData, target: Vec<f32>) {
        for i in 0..data.node_values.len() {
            let cost_derivative = self.cost.calculate_derivative(target[i], data.activations[i]);
            let activation_derivative = self.activation.calculate_derivative(data.weighted_inputs[i]);
            data.node_values[i] = cost_derivative * activation_derivative;
        }
    }

    pub fn calculate_hidden_layer_node_values(&self, data: LayerData, old_layer: Layer, old_node_values: Vec<f32>) {
        for i in 0..self.nodes_out {
            let mut new_node_value = 0.0f32;
            for j in 0..old_node_values.len() {
                let weighted_input_derivative = old_layer.weights[i as usize][j as usize];
                new_node_value += weighted_input_derivative * old_node_values[j];
            }
            new_node_value *= self.activation.calculate_derivative(data.weighted_inputs[i as usize]);
            data.node_values[i as usize] = new_node_value;
        }
    }

    pub fn update_gradients(&mut self, data: LayerData) {
        for i in 0..self.nodes_out {
            let node_value = data.node_values[i as usize];
            for j in 0..self.nodes_in {
                let derivative_cost_wrt_weight = data.inputs[j as usize] * node_value;
                self.cost_gradient_weights[i as usize][j as usize] += derivative_cost_wrt_weight;
            }
            let derivative_cost_wrt_bias = node_value;
            self.cost_gradient_biases[i as usize] += derivative_cost_wrt_bias;
        }
    }
}
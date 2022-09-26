use crate::Layer;
use crate::layer::LayerError;

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_sizes: &[u32]) -> Network {
        let mut layers: Vec<Layer> = vec![];
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Network { layers }
    }

    pub fn calculate_outputs(&self, inputs: &[f32]) -> Result<Vec<f32>, LayerError> {
        let mut outputs = inputs.to_vec();
        for layer in &self.layers {
            outputs = layer.calculate_outputs(&outputs)?;
        }
        Ok(outputs)
    }

    pub fn calculate_cost(&self, inputs: &[f32], targets: &[f32]) -> Result<f32, LayerError> {
        let outputs = self.calculate_outputs(inputs)?;
        let mut cost = 0.0;
        let output_layer = &self.layers[self.layers.len() - 1];
        for i in 0..outputs.len() {
            cost += output_layer.calculate_node_cost(targets[i], outputs[i]);
        }

        Ok(cost)
    }

    pub fn get_output_layer(&self) -> &Layer {
        &self.layers[self.layers.len() - 1]
    }

    pub fn get_input_layer(&self) -> &Layer {
        &self.layers[0]
    }

    pub fn get_hidden_layers(&self) -> &[Layer] {
        &self.layers[1..self.layers.len() - 1]
    }

    pub fn update_all_gradients(&mut self, inputs: &[f32], targets: &[f32]) -> Result<(), LayerError> {
        let outputs = self.calculate_outputs(inputs)?;
        let output_layer = &self.layers[self.layers.len() - 1];
        let node_values = output_layer.calculate_output_layer_node_values(targets);

        Ok(())
    }
}
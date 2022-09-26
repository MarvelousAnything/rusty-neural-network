use crate::{Layer, Network};


struct TrainingContext {
    pub network: Network,
    pub layer_data: Vec<LayerData>,
    pub data: DataSet
}

impl TrainingContext {
    pub fn new(network: Network, data: DataSet) -> TrainingContext {
        let layer_data = network.layers.iter().map(|layer| LayerData::new(layer)).collect();
        TrainingContext { network, layer_data, data}
    }

    pub fn train(&self, epochs: u32, learning_rate: f32) {
        for _ in 0..epochs {
            let current_data = self.data.select_random();
        }
    }

    pub fn update_gradients(&mut self, point: DataPoint) {
        let mut inputs_to_next_layer = point.inputs;
        for i in 0..self.network.layers.len() {
            inputs_to_next_layer = self.network.layers[i].calculate_training_outputs(inputs_to_next_layer.as_mut_slice(), self.layer_data[i]);
        }
    }
}

pub struct DataSet {
    pub points: Vec<DataPoint>
}

impl DataSet {
    pub fn new(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> DataSet {
        assert_eq!(inputs.len(), targets.len());
        DataSet {
            points: inputs.iter().zip(targets.iter()).map(|(input, target)| DataPoint {inputs: input.clone(), targets: target.clone()}).collect()
        }
    }

    pub fn select_random(&self) -> &DataPoint {
        let index = rand::random::<usize>() % self.points.len();
        &self.points[index]
    }
}

struct DataPoint {
    inputs: Vec<f32>,
    targets: Vec<f32>
}

impl DataPoint {
    pub fn new(inputs: Vec<f32>, targets: Vec<f32>) -> DataPoint {
        DataPoint { inputs, targets }
    }
}

pub struct LayerData {
    pub inputs: Vec<f32>,
    pub weighted_inputs: Vec<f32>,
    pub activations: Vec<f32>,
    pub node_values: Vec<f32>,
}

impl LayerData {
    pub fn new(layer: &Layer) -> LayerData {
        LayerData {
            inputs: vec![],
            weighted_inputs: vec![0.0; layer.nodes_out as usize],
            activations: vec![0.0; layer.nodes_out as usize],
            node_values: vec![0.0; layer.nodes_out as usize],
        }
    }
}

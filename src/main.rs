use std::ops::{Add, Neg};
use rand::{Rng, thread_rng};
use rand::distributions::Standard;

fn sigmoid(x: f64) -> f64 {
    x.neg().exp().add(1f64).powi(-1)
}

#[derive(Debug)]
struct Node {
    weight: Vec<f64>,
    bias: f64,
}

impl Node {
    pub fn new(prev_nodes: usize) -> Node {
        let weight: Vec<f64> = thread_rng().sample_iter(Standard).take(prev_nodes).collect();
        let bias: f64 = thread_rng().gen_range(0.0..1.0);
        Node { weight, bias }
    }
}

#[derive(Debug)]
struct Layer {
    nodes: Vec<Node>,
    layer_size: usize
}

impl Layer {
    pub fn new(layer_size: usize, prev_layer_size: usize) -> Layer {
        let nodes: Vec<Node> = (0..layer_size).map(|_| Node::new(prev_layer_size)).collect();
        Layer { nodes, layer_size }
    }
}


#[derive(Debug)]
struct Network {
    sizes: Vec<usize>,
    layers: Vec<Layer>,
    num_layers: usize
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Network {
        let mut layers: Vec<Layer> = Vec::new();
        let mut prev_size: usize = 0;
        for size in &sizes {
            let layer: Layer = Layer::new(*size, prev_size);
            prev_size = *size;
            layers.push(layer);
        }
        let num_layers = sizes.len();
        Network { sizes, layers, num_layers }
    }
}

/*
struct NetworkBuilder {
    inputs: usize,
    outputs: usize,
    hidden_layers: Vec<Layer>,
}

impl NetworkBuilder {
    pub fn new(inputs: usize, outputs: usize) -> NetworkBuilder {
        let hidden_layers: Vec<Layer> = Vec::new();
        NetworkBuilder { inputs, outputs, hidden_layers }
    }

    pub fn add_layer(mut self, layer_size: usize) -> NetworkBuilder {
        let layer = Layer::new(layer_size);
        self.hidden_layers.push(layer);
        self
    }

    pub fn build(self) -> Network {
        let input_layer: Layer = Layer::new(self.inputs);
        let output_layer: Layer = Layer::new(self.outputs);

        Network { inputs: self.inputs, outputs: self.outputs, input_layer, hidden_layers: self.hidden_layers, output_layer }
    }
}
*/

fn main() {
    let sizes = vec![2, 3, 1];
    let network = Network::new(sizes);
    println!("{:#?}", network)


}


use std::ops::{Add, Div, Mul, Neg};
use ndarray::{arr1, arr2, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;

fn sigmoid(x: f64) -> f64 {
    x.neg().exp().add(1.0).powi(-1)
}

fn sigmoid_derivative(x: f64) -> f64 {
    x.exp().div(x.exp().add(1.0).powi(2))
}

#[derive(Debug, Clone)]
struct Layer {
    size: usize,
    bias: f64,
    weights: Array2<f64>,
    outputs: Array1<f64>,
    deltas: Array1<f64>
}

impl Layer {
    pub fn new(size: usize, prev_size: usize) -> Layer {
        let bias: f64 = thread_rng().gen_range(0.0..1.0);
        let weights: Array2<f64> = Array2::random((prev_size, size), Uniform::new(0.0, 1.0));
        Layer { size, bias, weights, outputs: Array1::zeros(size), deltas: Array1::zeros(size) }
    }

    pub fn new_from_weights_and_bias(size: usize, weights: Array2<f64>, bias: f64) -> Layer {
        Layer { size, bias, weights, outputs: Array1::zeros(size), deltas: Array1::zeros(size) }
    }

    pub fn forward(&mut self, inputs: &Array1<f64>) {
        self.outputs = inputs.dot(&self.weights) + self.bias;
    }

    pub fn activate(&mut self) {
        self.outputs = self.outputs.map(|x| sigmoid(*x));
    }

    pub fn backward(&mut self, inputs: &Array1<f64>, next_layer: &Layer) {
        self.deltas = next_layer.weights.t().dot(&next_layer.deltas).map(|x| sigmoid_derivative(*x));
        self.weights = self.weights.map(|x| x - x.mul(0.1).mul(self.deltas.dot(inputs)));
        self.bias -= self.deltas.dot(inputs).mul(0.1);
    }

    pub fn set_weights(&mut self, weights: Array2<f64>) {
        self.weights = weights;
    }

    pub fn set_outputs(&mut self, outputs: Array1<f64>) {
        self.outputs = outputs;
    }

    pub fn set_deltas(&mut self, deltas: Array1<f64>) {
        self.deltas = deltas;
    }
}

#[derive(Debug)]
struct Network {
    topology: Vec<usize>,
    topology_size: usize,
    input_size: usize,
    output_size: usize,
    layers: Vec<Layer>
}

impl Network {
    pub fn new(topology: Vec<usize>) -> Network {
        let mut layers: Vec<Layer> = Vec::new();
        let topology_size = topology.len();
        for size in 0..topology_size-1 {
            layers.push(Layer::new(topology[size + 1], topology[size]));
        }
        let input_size: usize = topology[0];
        let output_size: usize = topology[topology.len() - 1];
        Network { topology, topology_size, input_size, output_size, layers }
    }

    pub fn new_with_weights_and_bias(topology: Vec<usize>, weights: Vec<Array2<f64>>, biases: Vec<f64>) -> Network {
        let mut layers: Vec<Layer> = Vec::new();
        let topology_size = topology.len();
        for size in 0..topology_size-1 {
            let weight = &weights[size];
            layers.push(Layer::new_from_weights_and_bias(topology[size + 1], weight.clone(), biases[size]));
        }
        let input_size: usize = topology[0];
        let output_size: usize = topology[topology.len() - 1];
        Network { topology, topology_size, input_size, output_size, layers }
    }

    pub fn think(&self, input_data: Array1<f64>) -> Vec<Array1<f64>> {
        let mut input: Array1<f64> = input_data;
        let mut outputs: Vec<Array1<f64>> = Vec::new();
        for layer in self.layers.iter() {
            input = layer.weights.t().dot(&input) + layer.bias;
            input = input.map(|x| sigmoid(*x));
            outputs.push(input.clone());
        }
        outputs
    }

    pub fn get_layer(&mut self, index: usize) -> &mut Layer {
        &mut self.layers[index]
    }

    pub fn mut_think(&mut self, input_data: Array1<f64>) {

        let mut input = input_data;

        for i in 0..&self.topology_size-1 {
            let layer = &mut self.layers[i];
            layer.forward(&input);
            input = layer.outputs.clone();
            println!("{:#?}", layer);
        }

    }

    pub fn train(&self, training_inputs: Array1<f64>, training_outputs: Array1<f64>, epochs: usize) {
        for _ in 0..epochs {
            let outputs = self.think(training_inputs.clone());
            let mut output_error: Array1<f64> = &training_outputs - &outputs[self.topology_size-2];

            // for i in (0..self.topology_size-1).rev() {
            //     let out = &outputs[i];
            //     let weights = &self.layers[i].weights;
            //     let gradients = out.map(|x| sigmoid_derivative(*x)) * &output_error;
            //
            //     println!("\n{:-^20}", self.topology_size-3 + i);
            //     println!("out: {:#?}", out);
            //     println!("weights: {:#?}", weights);
            //     println!("\ngradients: {gradients:#?}");
            //     println!("\nerror: {output_error:#?}");
            //     println!("\ntest: {:#?}", gradients.dot(&weights.t()));
            //
            //     output_error = outputs[i - 1].clone();
            // }

            // training_inputs.clone().dot(&(output.map(|x| sigmoid_derivative(*x)) * error));
        }
    }

    // pub fn backpropagate(&mut self, target: Array1<f64>, outputs: Vec<Array1<f64>>) {
    //     for i in (0..self.topology_size-1).rev() {
    //         let mut errors: Array1<f64>;
    //
    //         if i == self.topology_size - 2 {
    //             errors = &target - &outputs;
    //         }
    //     }
    // }
}

fn main() {
    // let mut network = Network::new(vec![2, 3, 2]);
    let inputs: Array1<f64> = arr1(&[0.1, 0.1]);
    let outputs: Array1<f64> = arr1(&[0.1, 0.1]);
    let weights: Vec<Array2<f64>> = vec![arr2(&[[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]), arr2(&[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])];
    let biases: Vec<f64> = vec![0., 0.];

    let mut network = Network::new_with_weights_and_bias(vec![2, 3, 2], weights, biases);


    println!("{:#?}", network);
    network.mut_think(inputs);
    println!("{:#?}", network);

    let mut layer = network.get_layer(0).clone();
    let layer2 = network.get_layer(1).clone();
    layer.backward(&layer2.outputs, &layer2);
    println!("{:#?}", layer);
    println!("{:#?}", layer2);

    // network.train(inputs, outputs, 1);
    // println!("{:#?}", network.think(inputs));
}
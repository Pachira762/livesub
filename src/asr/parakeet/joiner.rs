use candle::{Module, Result, Tensor};
use candle_nn::{Activation, Linear, Sequential, VarBuilder};

pub struct RnntJoiner {
    enc: Linear,
    pred: Linear,
    joint_net: Sequential,
}

impl RnntJoiner {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let encoder_hidden = 1024;
        let pred_hidden = 640;
        let joint_hidden = 640;
        let num_classes = 1024 + 1 + 5;

        let enc = candle_nn::linear(encoder_hidden, joint_hidden, vb.pp("enc"))?;
        let pred = candle_nn::linear(pred_hidden, joint_hidden, vb.pp("pred"))?;
        let joint_net = candle_nn::seq()
            .add(Activation::Relu)
            .add(candle_nn::linear(
                joint_hidden,
                num_classes,
                vb.pp("joint_net.2"),
            )?);

        Ok(Self {
            enc,
            pred,
            joint_net,
        })
    }

    pub fn joint(&self, f: &Tensor, g: &Tensor) -> Result<Tensor> {
        let f = self.enc.forward(f)?.unsqueeze(2)?;
        let g = self.pred.forward(g)?.unsqueeze(1)?;
        let x = (f + g)?;
        let x = self.joint_net.forward(&x)?;

        Ok(x)
    }
}

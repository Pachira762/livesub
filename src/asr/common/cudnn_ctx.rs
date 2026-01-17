use std::{cell::RefCell, collections::HashMap, sync::Arc};

use candle::{CudaDevice, Result, cuda::DeviceId};
use cudarc::cudnn::Cudnn;

thread_local! {
    static CUDNN: RefCell<HashMap<DeviceId, Arc<Cudnn>>> = HashMap::new().into();
}

pub fn get(device: &CudaDevice) -> Result<Arc<Cudnn>> {
    let device_id = device.id();
    CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(device.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        Ok(c?)
    })
}

#[allow(unused)]
pub fn shutdown() {
    CUDNN.with(|cudnn| {
        cudnn.borrow_mut().clear();
    });
}

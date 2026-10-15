import MetalPerformanceShadersGraph

extension MPSGraph {
    /// The compact model's original linear component and signed-difference
    /// objective. Every term reads original-grid pixels; targets are unchanged.
    func linearDetailLoss(prediction: MPSGraphTensor, target: MPSGraphTensor) -> MPSGraphTensor {
        let axes: [NSNumber] = [0, 1, 2, 3]
        let shape = prediction.shape!.map(\.intValue)
        let absolute = mean(of: absolute(with: subtraction(prediction, target, name: nil), name: nil), axes: axes, name: nil)
        var detail = scalar(0)
        func delta(_ value: MPSGraphTensor, axis: Int, offset: Int) -> MPSGraphTensor {
            var starts = [0, 0, 0, 0], ends = shape
            starts[axis] = offset
            let right = sliceTensor(value, starts: starts.map(NSNumber.init(value:)), ends: shape.map(NSNumber.init(value:)), strides: [1, 1, 1, 1], name: nil)
            ends[axis] -= offset
            let left = sliceTensor(value, starts: [0, 0, 0, 0], ends: ends.map(NSNumber.init(value:)), strides: [1, 1, 1, 1], name: nil)
            return subtraction(right, left, name: nil)
        }
        for offset in [1, 2, 4, 8] {
            for axis in [2, 3] where shape[axis] > offset {
                let error = subtraction(delta(prediction, axis: axis, offset: offset), delta(target, axis: axis, offset: offset), name: nil)
                let average = mean(of: self.absolute(with: error, name: nil), axes: axes, name: nil)
                detail = addition(detail, division(average, scalar(Double(offset)), name: nil), name: nil)
            }
        }
        return addition(absolute, scaled(detail, 4), name: nil)
    }
}

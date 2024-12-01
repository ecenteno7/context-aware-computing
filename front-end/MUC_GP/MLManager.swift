import TensorFlowLite
import CoreMotion

class MLManager {
    private var interpreter: Interpreter?
    
    // Window size for sensor data collection (how many samples we'll collect before making a prediction)
    private let windowSize = 50
    private var accelerometerWindow: [(x: Double, y: Double, z: Double)] = []
    private var gyroscopeWindow: [(x: Double, y: Double, z: Double)] = []
    private var audioWindow: [Double] = []
    
    init() {
        setupModel()
    }
    
    private func setupModel() {
        guard let modelPath = Bundle.main.path(forResource: "your_model", ofType: "tflite") else {
            print("Failed to load model")
            return
        }
        
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter?.allocateTensors()
            
            // Print input/output tensor details for debugging
            if let inputTensor = try? interpreter?.input(at: 0) {
                print("Input tensor shape: \(inputTensor.shape)")
                print("Input tensor type: \(inputTensor.dataType)")
            }
        } catch {
            print("Failed to create interpreter: \(error.localizedDescription)")
        }
    }
    
    // Called whenever new sensor data arrives
    func processSensorData(accelerometer: (x: Double, y: Double, z: Double),
                         gyroscope: (x: Double, y: Double, z: Double),
                         audioLevel: Double) {
        // Add new data to windows
        accelerometerWindow.append(accelerometer)
        gyroscopeWindow.append(gyroscope)
        audioWindow.append(audioLevel)
        
        // Keep only windowSize most recent samples
        if accelerometerWindow.count > windowSize {
            accelerometerWindow.removeFirst()
            gyroscopeWindow.removeFirst()
            audioWindow.removeFirst()
        }
        
        // If we have enough data, run inference
        if accelerometerWindow.count == windowSize {
            runInference()
        }
    }
    
    private func prepareInputData() -> [Float32]? {
        // Format the sensor data according to your model's input requirements
        // This is an example - adjust based on your model's specific needs
        var inputData: [Float32] = []
        
        // Add accelerometer data
        accelerometerWindow.forEach { accel in
            inputData.append(Float32(accel.x))
            inputData.append(Float32(accel.y))
            inputData.append(Float32(accel.z))
        }
        
        // Add gyroscope data
        gyroscopeWindow.forEach { gyro in
            inputData.append(Float32(gyro.x))
            inputData.append(Float32(gyro.y))
            inputData.append(Float32(gyro.z))
        }
        
        // Add audio data
        audioWindow.forEach { audio in
            inputData.append(Float32(audio))
        }
        
        return inputData
    }
    
    private func runInference() {
        guard let inputData = prepareInputData() else {
            print("Failed to prepare input data")
            return
        }
        
        do {
            // Convert Float32 array to Data
            let inputBytes = inputData.withUnsafeBufferPointer { Data(buffer: $0) }
            try interpreter?.copy(inputBytes, toInputAt: 0)
            try interpreter?.invoke()
            
            // Get output tensor and convert Data to Float32 array
            guard let outputTensor = try? interpreter?.output(at: 0) else {
                return
            }
            
            let outputSize = outputTensor.data.count / MemoryLayout<Float32>.stride
            var outputArray = [Float32](repeating: 0, count: outputSize)
            
            _ = outputArray.withUnsafeMutableBytes { ptr in
                outputTensor.data.copyBytes(to: ptr)
            }
            
            // Process the outputs
            interpretResults(outputArray)
            
        } catch {
            print("Failed to run inference: \(error.localizedDescription)")
        }
    }
    private func interpretResults(_ outputs: [Float32]) {
        // Example interpretation - adjust based on your model's output format
        // For example, if your model outputs class probabilities:
        let activities = ["walking", "running", "sitting", "standing"]
        if let maxIndex = outputs.indices.max(by: { outputs[$0] < outputs[$1] }) {
            let predictedActivity = activities[maxIndex]
            let confidence = outputs[maxIndex]
            print("Predicted activity: \(predictedActivity) with confidence: \(confidence)")
            
            // Here you could notify your UI or store the results
            NotificationCenter.default.post(
                name: Notification.Name("ActivityPredicted"),
                object: nil,
                userInfo: ["activity": predictedActivity, "confidence": confidence]
            )
        }
    }
}

import SwiftUI
import CoreMotion
import AVFoundation


struct AudioWaveformView: View {
    let amplitude: Double
    let maxHeight: CGFloat = 40
    @State private var previousAmplitudes: [Double] = Array(repeating: 0, count: 20) // Store history for smooth curve
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                let width = geometry.size.width
                let centerY = maxHeight / 2
                let pointWidth = width / CGFloat(previousAmplitudes.count - 1)
                
                // Start at the left edge at center height
                path.move(to: CGPoint(x: 0, y: centerY))
                
                // Create control points for the curve
                for i in 0..<previousAmplitudes.count - 1 {
                    let x1 = CGFloat(i) * pointWidth
                    let x2 = CGFloat(i + 1) * pointWidth
                    
                    let y1 = centerY - (CGFloat(previousAmplitudes[i]) * maxHeight)
                    let y2 = centerY - (CGFloat(previousAmplitudes[i + 1]) * maxHeight)
                    
                    let controlPoint1 = CGPoint(
                        x: x1 + (pointWidth * 0.5),
                        y: y1
                    )
                    
                    let controlPoint2 = CGPoint(
                        x: x2 - (pointWidth * 0.5),
                        y: y2
                    )
                    
                    path.addCurve(
                        to: CGPoint(x: x2, y: y2),
                        control1: controlPoint1,
                        control2: controlPoint2
                    )
                }
            }
            .stroke(Color.green, lineWidth: 2)
        }
        .frame(height: maxHeight)
        .onChange(of: amplitude) { oldValue, newValue in
            // Update amplitude history
            previousAmplitudes.removeFirst()
            previousAmplitudes.append(min(max(newValue * 0.8, 0), 1)) // Scale amplitude and clamp between 0 and 1
        }
    }
}

class PreviewSensorManager: SensorManager {
    override func startCollecting() {
        // Simulate sensor data updates
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.accelerometerData = (
                Double.random(in: -1...1),
                Double.random(in: -1...1),
                Double.random(in: -1...1)
            )
            self?.gyroscopeData = (
                Double.random(in: -1...1),
                Double.random(in: -1...1),
                Double.random(in: -1...1)
            )
            self?.audioLevel = Double.random(in: 0...0.5)
        }
    }
    
    override func stopCollecting() {
        // No need to stop anything in preview
    }
    
    override func checkPermissions() {
        // Always grant permissions in preview
        motionPermissionGranted = true
        microphonePermissionGranted = true
    }
    
    override func getRecordingURL() -> URL? {
        // Return nil for preview
        return nil
    }
}

class SensorManager: ObservableObject {
    private let motionManager = CMMotionManager()
    private let audioSession = AVAudioSession.sharedInstance()
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?
    private var audioFile: AVAudioFile?
    private var recordingURL: URL?
    private let mlManager = MLManager()
    
    @Published var accelerometerData: (x: Double, y: Double, z: Double) = (0, 0, 0)
    @Published var gyroscopeData: (x: Double, y: Double, z: Double) = (0, 0, 0)
    @Published var audioLevel: Double = 0
    @Published var motionPermissionGranted = false
    @Published var microphonePermissionGranted = false
    
    init() {
        setupAudioEngine()
        checkPermissions()
    }
    
    func checkPermissions() {
        // Motion permission (will be determined when we start using it)
        motionPermissionGranted = true
        
        // Updated microphone permission request
        if #available(iOS 17.0, *) {
            AVAudioApplication.requestRecordPermission { [weak self] granted in
                DispatchQueue.main.async {
                    self?.microphonePermissionGranted = granted
                }
            }
        } else {
            audioSession.requestRecordPermission { [weak self] granted in
                DispatchQueue.main.async {
                    self?.microphonePermissionGranted = granted
                }
            }
        }
    }
    
    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        inputNode = audioEngine?.inputNode
    }
    
    private func setupWAVRecording() throws {
        guard let inputNode = inputNode else { return }
        
        // Create a temporary URL for the WAV file
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        recordingURL = documentsPath.appendingPathComponent("recording_\(Date().timeIntervalSince1970).wav")
        
        guard let recordingURL = recordingURL else { return }
        
        // Setup the audio format for WAV
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: inputNode.outputFormat(forBus: 0).sampleRate,
            channels: 1,
            interleaved: false
        )!
        
        // Create the audio file
        audioFile = try AVAudioFile(
            forWriting: recordingURL,
            settings: format.settings
        )
    }
    
    func startCollecting() {
        // Start motion sensors
        if motionManager.isAccelerometerAvailable {
            motionManager.accelerometerUpdateInterval = 0.1
            motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, error in
                guard let data = data else { return }
                self?.accelerometerData = (data.acceleration.x, data.acceleration.y, data.acceleration.z)
                
                // Send to ML Manager only if we have all sensor data
                if let self = self {
                    self.mlManager.processSensorData(
                        accelerometer: self.accelerometerData,
                        gyroscope: self.gyroscopeData,
                        audioLevel: self.audioLevel
                    )
                }
            }
        }
        
        if motionManager.isGyroAvailable {
            motionManager.gyroUpdateInterval = 0.1
            motionManager.startGyroUpdates(to: .main) { [weak self] data, error in
                guard let data = data else { return }
                self?.gyroscopeData = (data.rotationRate.x, data.rotationRate.y, data.rotationRate.z)
                
                // Send to ML Manager only if we have all sensor data
                if let self = self {
                    self.mlManager.processSensorData(
                        accelerometer: self.accelerometerData,
                        gyroscope: self.gyroscopeData,
                        audioLevel: self.audioLevel
                    )
                }
            }
        }
        
        // Setup and start audio recording
        do {
            try setupWAVRecording()
            try audioSession.setCategory(.playAndRecord, mode: .default)
            try audioSession.setActive(true)
            
            guard let inputNode = inputNode,
                  let audioFile = audioFile else { return }
            
            let format = inputNode.outputFormat(forBus: 0)
            
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, time in
                guard let self = self else { return }
                
                // Write buffer to file
                try? self.audioFile?.write(from: buffer)
                
                // Calculate audio level for display
                let channelData = buffer.floatChannelData?[0]
                let frameLength = UInt32(buffer.frameLength)
                
                var sum: Float = 0
                for i in 0..<Int(frameLength) {
                    let sample = channelData?[i] ?? 0
                    sum += sample * sample
                }
                let rms = sqrt(sum / Float(frameLength))
                
                DispatchQueue.main.async {
                    self.audioLevel = Double(rms)
                    
                    // Send to ML Manager with latest sensor data
                    self.mlManager.processSensorData(
                        accelerometer: self.accelerometerData,
                        gyroscope: self.gyroscopeData,
                        audioLevel: self.audioLevel
                    )
                }
            }
            
            audioEngine?.prepare()
            try audioEngine?.start()
            
        } catch {
            print("Audio recording error: \(error.localizedDescription)")
        }
    }
    
    func stopCollecting() {
        motionManager.stopAccelerometerUpdates()
        motionManager.stopGyroUpdates()
        audioEngine?.stop()
        inputNode?.removeTap(onBus: 0)
        
        // Close the audio file
        if let recordingURL = recordingURL {
            print("Recording saved at: \(recordingURL.path)")
        }
    }
    
    func getRecordingURL() -> URL? {
        return recordingURL
    }
}

struct ActivityEntry: Identifiable {
    let id = UUID()
    let text: String
    let timestamp: TimeInterval
}

struct ActivityRecorderView: View {
    #if DEBUG
    @StateObject private var sensorManager = PreviewSensorManager()
    #else
    @StateObject private var sensorManager = SensorManager()
    #endif
    
    @State private var isRecording = false
    @State private var isCollectingData = false
    @State private var elapsedTime: TimeInterval = 0
    @State private var timer: Timer?
    @State private var activities: [ActivityEntry] = []
    @State private var showingPermissionAlert = false
    
    let sampleActivities = [
        "You are currently working remotely at a cafe",
        "You are currently studying in a library",
        "You are currently writing at home in a home office."
    ]
    
    var body: some View {
        VStack(spacing: 20) {
            Text("What am I currently doing?")
                .font(.title2)
                .fontWeight(.bold)
            
            if isCollectingData {
                Button(action: stopRecording) {
                    VStack {
                        Text("Collecting data...")
                            .foregroundColor(.white)
                        Text(timeString(from: elapsedTime))
                            .foregroundColor(.white)
                            .font(.subheadline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.orange)
                    .cornerRadius(10)
                }
            } else {
                Button(action: startRecordingWithPermissionCheck) {
                    Text(isRecording ? "New Recording" : "Start Recording")
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .cornerRadius(10)
                }
            }
            
            VStack(alignment: .leading, spacing: 12) {
                Text("Output")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                if isCollectingData {
                    outputView
                } else {
                    VStack(spacing: 0) {
                        Spacer()
                        Text("Start recording to detect your activity")
                            .foregroundColor(.secondary)
                            .italic()
                        Spacer()
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(height: UIScreen.main.bounds.height * 0.45)
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(.systemBackground))
                    .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color(.systemGray5), lineWidth: 1)
            )
            
            if isCollectingData {
                sensorDataView
            }
            
            Spacer()
        }
        .padding()
        .background(Color(.systemGray6).ignoresSafeArea())
        .alert("Permissions Required", isPresented: $showingPermissionAlert) {
            Button("Open Settings") {
                if let settingsUrl = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(settingsUrl)
                }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Microphone access is required to detect your activities. Please enable it in Settings.")
        }
    }
    
    private var outputView: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Circle()
                    .fill(Color.green)
                    .frame(width: 8, height: 8)
                Text("Active Recording")
                    .font(.subheadline)
                    .foregroundColor(.green)
            }
            
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(activities) { activity in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(activity.text)
                                .foregroundColor(.primary)
                            Text(timeString(from: activity.timestamp))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color(.systemGray6))
                        )
                    }
                }
            }
        }
    }
    
    private var sensorDataView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Sensor Data")
                .font(.headline)
                .foregroundColor(.primary)
            
            HStack {
                // Accelerometer
                VStack(alignment: .leading) {
                    Text("Accel")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.2f", sensorManager.accelerometerData.x))
                    Text(String(format: "%.2f", sensorManager.accelerometerData.y))
                    Text(String(format: "%.2f", sensorManager.accelerometerData.z))
                }
                
                Spacer()
                
                // Gyroscope
                VStack(alignment: .leading) {
                    Text("Gyro")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.2f", sensorManager.gyroscopeData.x))
                    Text(String(format: "%.2f", sensorManager.gyroscopeData.y))
                    Text(String(format: "%.2f", sensorManager.gyroscopeData.z))
                }
                
                Spacer()
                
                // Audio Level with waveform
                VStack(alignment: .leading) {
                    Text("Audio")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    AudioWaveformView(amplitude: sensorManager.audioLevel)
                        .frame(width: 60)
                }
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(.systemBackground))
                    .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
            )
        }
    }
    
    private func startRecordingWithPermissionCheck() {
        if !sensorManager.microphonePermissionGranted {
            showingPermissionAlert = true
            return
        }
        startRecording()
    }
    
    private func startRecording() {
        isRecording = true
        isCollectingData = true
        elapsedTime = 0
        activities.removeAll()
        sensorManager.startCollecting()
        
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            elapsedTime += 1
            
            // Wait 5 seconds before showing first activity
            if elapsedTime > 5 && Int(elapsedTime - 5) % 10 == 0 {  // Changed from % 3 to % 10
                let newActivity = ActivityEntry(
                    text: sampleActivities.randomElement() ?? "",
                    timestamp: elapsedTime
                )
                withAnimation {
                    activities.insert(newActivity, at: 0)
                }
            }
        }
        
//        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
//            elapsedTime += 1
//            
//            if Int(elapsedTime) % 3 == 0 {
//                let newActivity = ActivityEntry(
//                    text: sampleActivities.randomElement() ?? "",
//                    timestamp: elapsedTime
//                )
//                withAnimation {
//                    activities.insert(newActivity, at: 0)
//                }
//            }
//        }
    }
    
    private func stopRecording() {
        isCollectingData = false
        timer?.invalidate()
        timer = nil
        sensorManager.stopCollecting()
        
        // Get the URL of the recorded WAV file
        if let url = sensorManager.getRecordingURL() {
            print("Recording saved at: \(url.path)")
        }
    }
    
    private func timeString(from timeInterval: TimeInterval) -> String {
        let minutes = Int(timeInterval) / 60
        let seconds = Int(timeInterval) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
}

struct ActivityRecorderView_Previews: PreviewProvider {
    static var previews: some View {
        ActivityRecorderView()
    }
}

//
//  ViewController.swift
//  DangerZoneDetectorPrototype
//
//  Created by Gagson Lee on 2022/04/14.
//

import UIKit
import Metal
import MetalKit
import ARKit
import AVFoundation
import VideoToolbox

extension MTKView : RenderDestinationProvider {
}

class ViewController: UIViewController, MTKViewDelegate, ARSessionDelegate {
    
    @IBOutlet weak var infoLabel: UILabel!
    @IBOutlet weak var mainImageView: UIImageView!
    var bufferSize: CGSize!
    var session: ARSession!
    var renderer: Renderer!
    var mtkView: MTKView!
    var supportedVideoFormats: [ARConfiguration.VideoFormat]!
    public var recognizedImage : UIImage! = nil
    //    var lastOrientation: CGImagePropertyOrientation = .right
    let predictionQueue = DispatchQueue(label: "predictionQueue",
                                        qos: .userInitiated,
                                        attributes: [],
                                        autoreleaseFrequency: .inherit,
                                        target: nil)
    public let syncQueue = DispatchQueue(label: "RecognizationSyncQueue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    public lazy var speechSynthesizer = AVSpeechSynthesizer()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set the view's delegate
        session = ARSession()
        session.delegate = self
        
        // Set the view to use the default device
        if let view = self.view as? MTKView {
            view.device = MTLCreateSystemDefaultDevice()
            view.backgroundColor = UIColor.clear
            view.delegate = self
            guard view.device != nil else {
                print("Metal is not supported on this device")
                return
            }
//            view.preferredFramesPerSecond = 3 //Limit fps
            
            // Configure the renderer to draw to the view
            renderer = Renderer(session: session, metalDevice: view.device!, renderDestination: view)
            
            renderer.drawRectResized(size: view.bounds.size)
        }
        //
        //                let tapGesture = UITapGestureRecognizer(target: self, action: #selector(ViewController.handleTap(gestureRecognize:)))
        //                view.addGestureRecognizer(tapGesture)
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        //check lidar ability
        guard ARWorldTrackingConfiguration.supportsFrameSemantics([.smoothedSceneDepth]) else {
            print( "This device does not support LiDAR" )
            return
        }
        
        
        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = [.smoothedSceneDepth]
        
        //Setting video format as 1920x1080
        let selectedVideoFormat = ARWorldTrackingConfiguration.supportedVideoFormats[1]
        configuration.videoFormat = selectedVideoFormat
        print(selectedVideoFormat) //imageResolution=(1920, 1080) framesPerSecond=(60) for iPhone 12 Pro
        
        // Run the view's session
        session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        session.pause()
    }
    
    //    @objc
    //    func handleTap(gestureRecognize: UITapGestureRecognizer) {
    //        // Create anchor using the camera's current position
    //        if let currentFrame = session.currentFrame {
    //
    //            // Create a transform with a translation of 0.2 meters in front of the camera
    //            var translation = matrix_identity_float4x4
    //            translation.columns.3.z = -0.2
    //            let transform = simd_mul(currentFrame.camera.transform, translation)
    //
    //            // Add a new anchor to the session
    //            let anchor = ARAnchor(transform: transform)
    //            session.add(anchor: anchor)
    //        }
    //    }
    
    // MARK: - MTKViewDelegate
    
    // Called whenever view changes orientation or layout is changed
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.drawRectResized(size: size)
    }
    
    // Called whenever the view needs to render
    func draw(in view: MTKView) {
        renderer.update()
    }
    
    // MARK: - ARSessionDelegate
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user
        
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        // Inform the user that the session has been interrupted, for example, by presenting an overlay
        
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        // Reset tracking and/or remove existing anchors if consistent tracking is required
        
    }
    
    var frameCounter = 0
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if self.frameCounter > 0 {
            self.frameCounter -= 1
            return
        }
        self.frameCounter = 19 //fps = 3
        predictionQueue.async {
            if let currentFrame = self.session.currentFrame {
                let image = currentFrame.capturedImage
                let orientation = Int32(CGImagePropertyOrientation.right.rawValue) //flip back the image into correct orientation
                let rawCiimage = CIImage(cvPixelBuffer: image)
                let ciimage = rawCiimage.oriented(forExifOrientation: orientation)
                let uiimage = UIImage(ciImage: ciimage)
                
                detectorMain(uiimage: uiimage, ciimage: ciimage, viewController: self, stage: 1, currentFrame: currentFrame) //commit object detection
            }
        }
    }
    
} //viewController

extension UIImage {     // for tranforming CVPixelBuffer to UIImage
    public convenience init?(pixelBuffer: CVPixelBuffer) {
        var cgImage: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cgImage)
        
        guard let cgImage = cgImage else {
            return nil
        }
        self.init(cgImage: cgImage)
    }
}

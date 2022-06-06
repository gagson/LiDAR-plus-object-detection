//
//  DetectorMain.swift
//  DangerZoneDetectorPrototype
//
//  Created by Lee on 2022/03/07.
//

import Foundation
import CoreML
import Vision
import UIKit
import ARKit

class detectorMain {
    
    var originalUIImage : UIImage
    var imageInProcess : CIImage
    var viewController : ViewController!
    var stage : Int
    var height, width : Int
    var ratio : Double
    var fullScale : Double
    var offset_x, offset_y : Int
    var currentFrame : ARFrame
    
    let confidenceTh = 0.0
    //    let anchorBoxes = [ [30.0,61.0], [62.0,45.0], [59.0,119.0], [116.0,90.0], [156.0,198.0], [373.0,326.0] ]
    let anchorBoxes = [ [116.0,90.0], [156.0,198.0], [373.0,326.0] , [62.0,45.0], [59.0,119.0], [116.0,90.0] ]
    let numClasses = 9
    
    init(uiimage : UIImage, ciimage: CIImage, viewController controller : ViewController, stage : Int, currentFrame: ARFrame) {
        self.originalUIImage = uiimage
        self.viewController = controller
        self.stage = stage
        self.currentFrame = currentFrame
        
        self.imageInProcess = ciimage
        self.height = Int(uiimage.size.height)
        self.width = Int(uiimage.size.width)
        //print("height=", self.height, "width=", self.width)
        if self.height > self.width {   // portrait
            self.ratio = Double(self.height) / 416.0
            self.fullScale = Double(self.height)
            self.offset_x = -(self.height-self.width)/2
            self.offset_y = 0
        } else {                        // landscape
            self.ratio = Double(self.width) / 416.0
            self.fullScale = Double(self.width)
            self.offset_x = 0
            self.offset_y = -(self.width-self.height)/2
        }
        exeDetection(for: ciimage)
    }
    
    func exeDetection(for image: CIImage) {
        let orientation: CGImagePropertyOrientation  = .up // up
        DispatchQueue.global(qos: .userInitiated).async {  // ASYNC execution
            // DispatchQueue.global(qos: .userInitiated).sync {   // for CPU time measure
            let handler = VNImageRequestHandler(ciImage: image, orientation: orientation)
            do {
                try handler.perform([self.detectionRequest])    // prediction task
            } catch {
                print("Failed to perform detection")
            }
        }
    }
    
    lazy var detectionRequest: VNCoreMLRequest = {  // prediction task
        do {
            let model = try VNCoreMLModel(for: DangerZones20220323().model)
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in self?.postCalculation(for: request, error: error) })       // detection by Yolo3Tiny is executed and then "postCalculation" is called
            request.imageCropAndScaleOption = .scaleFit
            return request
        } catch {
            fatalError("Failed to load Vision ML model: ¥(error)")
        }
    }()
    
    func postCalculation(for request: VNRequest, error: Error?){
        DispatchQueue.main.async {
            guard let results = request.results else {
                return
            }
            let features = results as! [VNCoreMLFeatureValueObservation]
            let rawObjs = self.decodeYoloOutput(for: features)
            let objs = self.clipBB( rawObjs: rawObjs )
            var closestObjLidar = closestCoorWithLidar( x:0, y:0, lidar:100.0 )
            var closestObj = detectedObject()
            for obj in objs {
                let detectedConfi = round(obj.confidence * 100)/100.0
                if detectedConfi >= 0.7 {
                    var labelName : String
                    switch obj.label {
                    case 0: labelName = "downstairEdgeC"
                    case 1: labelName = "downstairEdgeL"
                    case 2: labelName = "downstairEdgeR"
                    case 3: labelName = "platformDoor"
                    case 4: labelName = "platformEdgeC"
                    case 5: labelName = "platformEdgeL"
                    case 6: labelName = "platformEdgeR"
                    case 7: labelName = "trainOtherSide"
                    case 8: labelName = "trainSameSide"
                    default : labelName = ""
                    }
                    let closestLidarData = self.defineClosestArea(obj: obj) //get closest x, y, lidar
                    if closestLidarData.lidar > 0 && closestLidarData.lidar < 5 {//Commit detected information to viewController
                        if closestLidarData.lidar > 0 && closestLidarData.lidar <= 3 { //produce output for edges within 3m only
                            //if the object is a downstairEdge or platformEdge
                            if obj.label == 0 || obj.label == 1 || obj.label == 2 || obj.label == 4 || obj.label == 5 || obj.label == 6 {
                                if closestObjLidar.lidar > closestLidarData.lidar {
                                    closestObjLidar = closestLidarData
                                    closestObj = obj
                                }
                            }
                        }// >0&&<=3
                        self.viewController.syncQueue.sync {
                            self.viewController.infoLabel.text = "\(labelName) \(detectedConfi*100)% in \(closestLidarData.lidar)m"
                        }
                    }// >0&&<5
                    print("\(labelName) \(detectedConfi*100)% in \(closestLidarData)m")
                }
            } //for obj in objs
            if closestObjLidar.lidar <= 3.0 {
                self.initSpeech(obj: closestObj, coorWithLidar: closestObjLidar)
            }
            
            let uiimage = self.drawObj(ciimage : self.imageInProcess, objs : objs, width: self.width, height: self.height)
            // commit resulted uiimage to ViewController
            self.viewController.syncQueue.sync {
                self.viewController.recognizedImage = uiimage
                self.viewController.mainImageView.image = self.viewController.recognizedImage
            }
        }
    }
    func sigmoid(_ z : Double) -> Double {
        return 1.0 / (1.0 + exp(-z))
    }
    
    var anchor_for_debug : Int = 0
    
    func decodeYoloOutput(for features : [VNCoreMLFeatureValueObservation]) -> [detectedObject] {
        var detectedObjects : [detectedObject] = [] //the list of results to be returned
        // post process start
        for l in 0 ..< 2 { // for yolo1 and yolo2 outputs
            let yolo_out = features[l].featureValue.multiArrayValue! // MLMultiArray
            let out = UnsafeMutablePointer<Double>(OpaquePointer(yolo_out.dataPointer)) // #ch x Rows x Cols
            let Rows = Int(truncating: yolo_out.shape[1])
            let Cols = Int(truncating: yolo_out.shape[2])
            let anchorStride = Int(truncating: yolo_out.strides[0])
            let rowStride = Int(truncating: yolo_out.strides[1])
            
            // Extract high confidence boxes
            // print("new image")
            for row in 0 ..< Rows {
                for col in 0 ..< Cols {
                    let gridBase = row*rowStride + col
                    for anchor in 0 ..< 3 {
                        let index = gridBase + anchor*(5+numClasses)*anchorStride
                        let bconf = out[index+4*anchorStride]
                        if bconf > confidenceTh { // Now found true box!!
                            // print("boxConf="+String(out[index+4*anchorStride]))
                            
                            let boxConfidence = sigmoid(bconf)
                            
                            // decode class
                            var label = -1
                            var maxConf = -10000.0
                            for lb in 0 ..< numClasses {
                                let idx = lb + 5
                                let conf = out[index + idx*anchorStride ]
                                if conf > maxConf {
                                    label = lb
                                    maxConf = conf
                                }
                            }
                            if label < 0 || maxConf < confidenceTh {
                                break   // no valid object found
                            }
                            let classConfidence = sigmoid(maxConf)
                            //                            print(String(boxConfidence) + " " + String(label) + " " + String(classConfidence))
                            
                            // decode position and size of bounding box
                            let anchorBox = anchorBoxes[l*3+anchor]
                            self.anchor_for_debug = l*3+anchor
                            let tx = out[index]
                            let ty = out[index+anchorStride]
                            let tw = out[index+2*anchorStride]
                            let th = out[index+3*anchorStride]
                            let box_x = Int((sigmoid(tx)+Double(col))/Double(Cols)*self.fullScale)+self.offset_x
                            let box_y = Int((sigmoid(ty)+Double(row))/Double(Rows)*self.fullScale)+self.offset_y
                            let box_w = Int(exp(tw)*anchorBox[0]*self.ratio)
                            let box_h = Int(exp(th)*anchorBox[1]*self.ratio)
                            
                            //
                            let totalConfidence = boxConfidence * classConfidence
                            //print("Label:", label,
                            //        " confidence:", totalConfidence,
                            //        " (x,y)=(", box_x, "," , box_y, ")",
                            //        " (w,h)=(", box_w, "," , box_h, ")")
                            let new = detectedObject(label: label, confidence: totalConfidence, x:box_x, y:box_y, w:box_w, h:box_h)
                            
                            // check overlapped object
                            var addNew : Bool = true
                            var removeOld : Bool = false
                            var removeIndex : Int = -1
                            var index : Int = 0
                            for old in detectedObjects {
                                if old.overlaps(other : new) {
                                    if new.confidence > old.confidence {    // new one is better
                                        removeOld = true
                                        removeIndex = index
                                        break
                                    } else {                                // old one is better
                                        addNew = false
                                    }
                                }
                                index += 1
                            }
                            if removeOld {
                                detectedObjects.remove(at: removeIndex)
                            }
                            if addNew {
                                detectedObjects.append(new)
                            }
                        }
                    }
                }
            }
        }
        // post process done
        return detectedObjects
    }
    
    func drawObj(ciimage : CIImage, objs : [detectedObject], width : Int, height : Int) -> UIImage {
        
        let cont = CIContext(options:nil)
        let cgimage : CGImage = cont.createCGImage(ciimage, from: ciimage.extent)!
        // create CGcontext
        UIGraphicsBeginImageContext(CGSize(width: width, height: height))
        let context:CGContext = UIGraphicsGetCurrentContext()!
        context.setLineWidth(10.0)                   // set line width
        
        // draw image
        context.translateBy(x:0.0, y:CGFloat(height))    // move origin point to upper left corner
        context.scaleBy(x:1.0, y:-1.0)                        // reverse y-axis
        context.draw(cgimage, in: CGRect(x:0, y:0, width : width, height : height), byTiling:false)
        
        for obj in objs {
            let detectedConfi = round(obj.confidence * 100)/100.0
            if detectedConfi >= 0.7 {
                
                context.setStrokeColor(UIColor.red.cgColor)
                context.beginPath()
                let y1 = height - obj.y1  // (0,0) point of Quartz graphics is left bottom corner
                let y2 = height - obj.y2
                
                context.addRect(CGRect(x: obj.x1, y: y2, width: obj.x2-obj.x1, height: y1-y2))
                
                context.closePath()
                context.strokePath()
            }//if detectedConfi >= 0.7
        }
        let cgimage2 : CGImage = context.makeImage()!
        let uiimage : UIImage = UIImage(cgImage: cgimage2)
        UIGraphicsEndImageContext()
        return uiimage
    }
    
    struct closestCoorWithLidar {
        let x: Int
        let y: Int
        let lidar: Float
        init( x:Int, y:Int, lidar:Float ) {
            self.x = x
            self.y = y
            self.lidar = lidar
        }
    }
    
    func defineClosestArea(obj: detectedObject) -> closestCoorWithLidar {
        var areaList: [closestCoorWithLidar] = []
        
        switch obj.label {
        case 1, 5: //platformEdgeL or downstairEdgeL
            for index in 0...10 {
                let x = ((obj.x2 - obj.x1)*index)/10 + obj.x1
                let y = ((obj.y2 - obj.y1)*index)/10 + obj.y1
                let lidar = getDepth(x: Int(x), y: Int(y))
                let lidarM = Float(lidar)/100
                areaList.append(closestCoorWithLidar(x: Int(x), y: Int(y), lidar: lidarM))
            }
            let minLidar: Float = areaList.min(by: { $0.lidar < $1.lidar })?.lidar ?? -1.0
            let closestAreaIndex = areaList.firstIndex(where: { $0.lidar == minLidar })!
            let closestArea = areaList[closestAreaIndex]
            print(areaList)
            return closestArea
        case 2, 6: //platformEdgeR or downstairEdgeR
            for index in 0...10 {
                let x = ((obj.x2 - obj.x1)*index)/10 + obj.x1
                let y = obj.y2 - ((obj.y2 - obj.y1)*index)/10
                let lidar = getDepth(x: Int(x), y: Int(y))
                let lidarM = Float(lidar)/100
                areaList.append(closestCoorWithLidar(x: Int(x), y: Int(y), lidar: lidarM))
            }
            let minLidar: Float = areaList.min(by: { $0.lidar < $1.lidar })?.lidar ?? -1.0
            let closestAreaIndex = areaList.firstIndex(where: { $0.lidar == minLidar })!
            let closestArea = areaList[closestAreaIndex]
            print(areaList)
            return closestArea
        case 0, 3, 4, 7, 8, 9: //Other objects
            let x = (obj.x2 + obj.x1)/2
            let y = (obj.y2 + obj.y1)/2
            let lidar = getDepth(x: x, y: y)
            let lidarM = Float(lidar)/100
            return closestCoorWithLidar(x: x, y: y, lidar: lidarM)
        default:
            return closestCoorWithLidar(x: -1, y: -1, lidar: -1)
        }
    }
    
    func getDepth(x:Int, y:Int) -> Int {
        
        let xLidar = Int(floor(Double(y))/7.5)
        let yLidar = Int(floor((1079.0-Double(x)))/7.5)
        
        //remove comment-out if lidar confidence data become necessary
        //        guard let confidenceMap = self.currentFrame.smoothedSceneDepth?.confidenceMap else { return -1 }
        //        CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)// CPU allocation for confidenceMap
        //        let baseConf = CVPixelBufferGetBaseAddress(confidenceMap)
        //        let widthConf = CVPixelBufferGetWidth(confidenceMap)
        //        let heightConf = CVPixelBufferGetHeight(confidenceMap)
        //
        //        let bindPtrConf = baseConf?.bindMemory(to: UInt8.self, capacity: widthConf * heightConf)
        //        let bufPtrConf = UnsafeBufferPointer(start: bindPtrConf, count: widthConf * heightConf)
        //        let confiArray = Array(bufPtrConf)
        //        CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) //Release allocated CPU
        //
        //        let fixedLidarConfiData = confiArray[256*yLidar+xLidar] //width*y+x, divide Mid point coordinate by 7.5 to get the pixel of lidar grid
        //        let lidarConfiData = Int(fixedLidarConfiData)
        //        if lidarConfiData > 0 {
        
        //Get lidar data of an object
        guard let depthMap = self.currentFrame.smoothedSceneDepth?.depthMap else { return -2 }
        CVPixelBufferLockBaseAddress(depthMap, .readOnly) // CPU allocation for depthMap
        let base = CVPixelBufferGetBaseAddress(depthMap)
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        
        let bindPtr = base?.bindMemory(to: Float32.self, capacity: width * height)
        let bufPtr = UnsafeBufferPointer(start: bindPtr, count: width * height)
        let depthArray = Array(bufPtr)
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) //Release allocated CPU
        
        let fixedArray = depthArray.map({ $0.isNaN ? 0 : $0 })
        
        let fixedLidarData = fixedArray[256*yLidar+xLidar] //width*y+x, divide Mid point coordinate by 7.5 to get the pixel of lidar grid
        let lidarData = Int(fixedLidarData*100.0) //metre -> centimetre
        
        return lidarData
        //        } //if lidarConfiData > 0
        //        return -3
    }
    
    var voice : AVSpeechSynthesisVoice!
    var leftsideWarning: AVSpeechUtterance!
    var rightsideWarning: AVSpeechUtterance!
    var frontsideWarning: AVSpeechUtterance!
    
    func initSpeech(obj: detectedObject, coorWithLidar: closestCoorWithLidar) {
        //2.2s per speech
        let fastRate : Float = (AVSpeechUtteranceMaximumSpeechRate + AVSpeechUtteranceDefaultSpeechRate*25.0) / 21.0
        let postDelay : Double = 0.0
        voice = AVSpeechSynthesisVoice(language: "ja-JP")
        var labelName : String
        switch obj.label {
        case 0, 1, 2: labelName = "下り階段"
        case 4, 5, 6: labelName = "ホーム端"
        case 3, 7, 8: labelName = "エラー１"
        default : labelName = "エラー２"
        }
        //closest point
        let x = coorWithLidar.x
        if x > 0 && x <= 360 {
            leftsideWarning = AVSpeechUtterance(string: "左前\(coorWithLidar.lidar)メートルに\(labelName)があります")
            leftsideWarning.voice = self.voice
            leftsideWarning.postUtteranceDelay = postDelay
            leftsideWarning.rate = fastRate
            print("left")
            self.produceSpeech(leftsideWarning, lidar: coorWithLidar.lidar)
        } else if x > 360 && x <= 720 {
            rightsideWarning = AVSpeechUtterance(string: "正面\(coorWithLidar.lidar)メートルに\(labelName)があります")
            rightsideWarning.voice = self.voice
            rightsideWarning.postUtteranceDelay = postDelay
            rightsideWarning.rate = fastRate
            print("central")
            self.produceSpeech(rightsideWarning, lidar: coorWithLidar.lidar)
        } else if x > 720 && x < 1080 {
            frontsideWarning = AVSpeechUtterance(string: "右前\(coorWithLidar.lidar)メートルに\(labelName)があります")
            frontsideWarning.voice = self.voice
            frontsideWarning.postUtteranceDelay = postDelay
            frontsideWarning.rate = fastRate
            print("right")
            self.produceSpeech(frontsideWarning, lidar: coorWithLidar.lidar)
        }
    }
    
    private func produceSpeech(_ utterance : AVSpeechUtterance, lidar : Float) {
        if self.viewController.speechSynthesizer.isSpeaking == false {
            self.viewController.speechSynthesizer.speak(utterance)
            return
        }
    }
    
    func clipBB( rawObjs: [detectedObject] ) -> [detectedObject] {
        var objs : [detectedObject] = []
        for rawObj in rawObjs {
            if rawObj.x1 >= 0 && rawObj.y1 >= 0 && rawObj.x2 < 1080 && rawObj.y2 < 1920 {
                // no need to clip
                objs.append( rawObj )
                continue
            }
            
            var obj = detectedObject()
            obj = rawObj
            let flag:Bool
            var box = [ rawObj.x1, rawObj.y1, rawObj.x2, rawObj.y2 ]
            switch rawObj.label {
            case 1, 5: //platformEdgeL or downstairEdgeL
                (flag, box) = self.clipLline( box:box, w: 1080, h: 1920 )
            case 2, 6: //platformEdgeR or downstairEdgeR
                (flag, box) = self.clipRline( box:box, w: 1080, h:1920 )
            case 0, 3, 4, 7, 8, 9: //Other objects
                (flag, box) = self.simpleClip( box:box, w:1080, h:1920)
            default:
                flag = false
            }
            if flag {
                obj.x1 = box[0]
                obj.y1 = box[1]
                obj.x2 = box[2]
                obj.y2 = box[3]
            }
        }
        return objs
    }
    
    func simpleClip( box:[Int], w:Int, h:Int ) -> (Bool,[Int]) {
        var clipped = box
        if clipped[0] >= w || clipped[2] < 0 || clipped[1] >= h || clipped[3] < 0 {
            return (false, clipped)
        }
        if clipped[0] < 0 {
            clipped[0] = 0;
        }
        if clipped[1] < 0 {
            clipped[1] = 0;
        }
        if clipped[2] >= w {
            clipped[2] = w-1;
        }
        if clipped[3] >= h {
            clipped[3] = h-1;
        }
        return (true,clipped)
    }
    
    func getX(x: Int, A: [Int], B: [Int]) -> Int {
        let a = Double(B[1]-A[1])/Double(B[0]-A[0])
        return Int(a*Double(x-A[0]))+A[1]
    }
    
    func getY(y: Int, A: [Int], B: [Int]) -> Int {
        let b = Double(B[0]-A[0])/Double(B[1]-A[1])
        return Int(b*Double(y-A[1]))+A[0]
    }
    
    func clipRline(box: [Int], w: Int, h: Int) -> (Bool, [Int]) {
        if box[0]>=0 && box[1]>=0 && box[2]<w && box[3]<h { //completely inside
            return (true, box)
        }
        if box[0]>=w || box[1]>=h || box[2]<0 || box[3]<0 { //completely outside
            return (false, box)
        }
        var A : [Int] = [box[2], box[1]] //x2,y1
        var B : [Int] = [box[0], box[3]] //x1,y2
        let IB = A[1]>=0    // Inside of Bottom edge
        let IR = A[0]<w     // Inside of Right edge
        let IL = B[0]>=0    // Inside of Left edge
        let IT = B[1]<h     // Inside of Top edge
        
        if IL && IT {
            if IR {
                A = [self.getX(x: 0,A: A,B: B), 0]
            } else if IB {
                A = [w-1, getY(y: w-1,A: A,B: B)]
            } else {
                let x = self.getX(x: 0, A: A, B: B)
                if x < w {
                    A = [x, 0]
                } else {
                    A = [w-1, getY(y: w-1, A: A, B: B)]
                }
            }
        } else if IR && IB {
            if IL {
                B = [getX(x: h-1,A: A,B: B), h-1]
            } else if IT {
                B = [0, getY(y: 0,A: A,B: B)]
            } else {
                let x = getX(x: h-1,A: A,B: B)
                if x >= 0 {
                    B = [x,h-1]
                } else {
                    B = [0,getY(y: 0,A: A,B: B)]
                }
            }
        } else if IB && IT {
            let ya = getY(y: w-1,A: A,B: B)
            let yb = getY(y: 0,A: A,B: B)
            A = [w-1, ya]
            B = [0, yb]
        } else if IL && IR {
            let xa = getX(x: 0,A: A,B: B)
            let xb = getX(x: h-1,A: A,B: B)
            A = [xa, 0]
            B = [xb, h-1]
        } else if !IL && !IB {
            let x = getX(x: 0,A: A,B: B)
            let y = getY(y: 0,A: A,B: B)
            if x >= 0 && y >= 0 {
                A = [x, 0]
                B = [0, y]
            } else {
                return (false, box)
            }
        } else if !IR && !IT {
            let x = getX(x: h-1,A: A,B: B)
            let y = getY(y: w-1,A: A,B: B)
            if x<w && y<h {
                A = [w-1,y]
                B = [x,h-1]
            } else {
                return (false, box)
            }
        } else {
            print( "error" )
        }
        let box = [B[0],A[1],A[0],B[1]]
        return (true, box)
    } //clipRline
    
    func clipLline(box: [Int], w: Int, h: Int) -> (Bool, [Int]) {// completely inside
        if box[0]>=0 && box[1]>=0 && box[2]<w && box[3]<h {
            return (true, box)
        }
        if box[0]>=w || box[1]>=h || box[2]<0 || box[3]<0 { //completely outside
            return (false, box)
        }
        var A : [Int] = [box[0], box[1]] // x_min, y_min
        var B : [Int] = [box[2], box[3]] // x_max, y_max
        let IB = A[1]>=0    // Inside of Bottom edge
        let IR = B[0]<w     // Inside of Right edge
        let IL = A[0]>=0    // Inside of Left edge
        let IT = B[1]<h     // Inside of Top edge
        if IR && IT {
            if IL {
                A = [getX(x: 0,A: A,B: B),0]
            } else if IB {
                A = [0,getY(y: 0,A: A,B: B)]
            } else {
                let x = getX(x: 0,A: A,B: B)
                if x >= 0 {
                    A = [x,0]
                } else {
                    A=[0,getY(y: 0,A: A,B: B)]
                }
            }
        } else if IL && IB {
            if IR {
                B = [getX(x: h-1,A: A,B: B), h-1]
            } else if IT {
                B = [w-1, getY(y: w-1,A: A,B: B)]
            } else {
                let x = getX(x: h-1,A: A,B: B)
                if x < w {
                    B = [x, h-1]
                } else {
                    B = [w-1, getY(y: w-1,A: A,B: B)]
                }
            }
        } else if IB && IT {
            let ya = getY(y: 0,A: A,B: B)
            let yb = getY(y: w-1,A: A,B: B)
            A = [0, ya]
            B = [w-1, yb]
        } else if IL && IR {
            let xa = getX(x: 0,A: A,B: B)
            let xb = getX(x: h-1,A: A,B: B)
            A = [xa, 0]
            B = [xb, h-1]
        } else if !IR && !IB {
            let x = getX(x: 0,A: A,B: B)
            let y = getY(y: w-1,A: A,B: B)
            if x<w && y>=0 {
                A = [x, 0]
                B = [w-1, y]
            } else {
                return (false, box)
            }
        } else if !IL && !IT {
            let x = getX(x: h-1,A: A,B: B)
            let y = getY(y: 0,A: A,B: B)
            if x<w && y<h {
                A = [0, y]
                A = [x, h-1]
            } else {
                return (false, box)
            }
        } else {
            print( "error" )
        }
        let box = [A[0],A[1],B[0],B[1]]
        return (true, box)
    } //clipLline
    
}

class detectedObject {
    public var label, x1, y1, x2, y2 : Int  // x1 <= x2 and y1 <= y2
    public var confidence : Double
    
    init() {
        self.label = 0
        self.confidence = 0.0
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
    }
    
    init(label : Int, confidence : Double, x : Int, y : Int, w : Int, h : Int) {
        self.label = label
        self.confidence = confidence
        let half_w = Int(w/2)
        let half_h = Int(h/2)
        self.x1 = x-half_w
        self.y1 = y-half_h
        self.x2 = x+half_w
        self.y2 = y+half_h
    }
    
    
    public func addOffset(offsetX : Int, offsetY : Int) {
        self.x1 += offsetX
        self.x2 += offsetX
        self.y1 += offsetY
        self.y2 += offsetY
    }
    
    public func copy(from : detectedObject) {
        self.label = from.label
        self.x1 = from.x1
        self.x2 = from.x2
        self.y1 = from.y1
        self.y2 = from .y2
        self.confidence = from.confidence
    }
    
    func overlaps(other : detectedObject) -> Bool {
        if self.label != other.label { return false }
        if self.x1 > other.x2 { return false }
        if self.x2 < other.x1 { return false }
        if self.y1 > other.y2 { return false }
        if self.y2 < other.y1 { return false }
        return true
    }
    
}

import cv2
import argparse
import json
import os.path
import ultimateAlprSdk


# https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html
JSON_CONFIG = {
    "debug_level": "info",
    "debug_write_input_image_enabled": False,
    "debug_internal_data_path": ".",
    
    "num_threads": -1,
    "gpgpu_enabled": True,
    "max_latency": -1,

    "klass_vcr_gamma": 1.5,
    
    "detect_roi": [0, 0, 0, 0],
    "detect_minscore": 0.1,

    "car_noplate_detect_min_score": 0.8,
    
    "pyramidal_search_enabled": True,
    "pyramidal_search_sensitivity": 0.28,
    "pyramidal_search_minscore": 0.3,
    "pyramidal_search_min_image_size_inpixels": 800,
    
    "recogn_rectify_enabled": True,
    "recogn_minscore": 0.3,
    "recogn_score_type": "min"
}


IMAGE_TYPES_MAPPING = { 
        'RGB': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGB24,
        'RGBA': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGBA32,
        'L': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_Y
}


def checkResult(operation, result):
    if not result.isOK():
        print(operation + ": FAILED -> " + result.phrase())
        assert False
    else:
        print(operation + ": OK -> " + result.json())


def create_parser():
    parser = argparse.ArgumentParser(description="""This is the recognizer sample using python language.""")

    parser.add_argument("--video_path", required=True,
                        help="Path to the image with ALPR data to recognize")
    parser.add_argument("--resolution", required=False, default="1920x1080",
                        help="Resolution of the image/video frame")
    parser.add_argument("--assets", required=False, default="../../../assets",
                        help="Path to the assets folder")
    parser.add_argument("--charset", required=False, default="latin",
                        help="Defines the recognition charset (a.k.a alphabet) value (latin, korean, chinese...)")
    parser.add_argument("--car_noplate_detect_enabled", required=False, default=False,
                        help="Whether to detect and return cars with no plate")
    parser.add_argument("--ienv_enabled", required=False, default=False,
                        help="Whether to enable Image Enhancement for Night-Vision (IENV). More info about IENV at https://www.doubango.org/SDKs/anpr/docs/Features.html#image-enhancement-for-night-vision-ienv. Default: true for x86-64 and false for ARM.")
    parser.add_argument("--openvino_enabled", required=False, default=True,
                        help="Whether to enable OpenVINO. Tensorflow will be used when OpenVINO is disabled")
    parser.add_argument("--openvino_device", required=False, default="CPU",
                        help="Defines the OpenVINO device to use (CPU, GPU, FPGA...). More info at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html#openvino-device")
    parser.add_argument("--npu_enabled", required=False, default=True,
                        help="Whether to enable NPU (Neural Processing Unit) acceleration")
    parser.add_argument("--klass_lpci_enabled", required=False, default=False,
                        help="Whether to enable License Plate Country Identification (LPCI). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#license-plate-country-identification-lpci")
    parser.add_argument("--klass_vcr_enabled", required=False, default=False,
                        help="Whether to enable Vehicle Color Recognition (VCR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-color-recognition-vcr")
    parser.add_argument("--klass_vmmr_enabled", required=False, default=False,
                        help="Whether to enable Vehicle Make Model Recognition (VMMR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-make-model-recognition-vmmr")
    parser.add_argument("--klass_vbsr_enabled", required=False, default=False,
                        help="Whether to enable Vehicle Body Style Recognition (VBSR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-body-style-recognition-vbsr")
    parser.add_argument("--tokenfile", required=False, default="",
                        help="Path to license token file")
    parser.add_argument("--tokendata", required=False, default="",
                        help="Base64 license token data")

    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()

    if not os.path.exists(args.video_path):
        raise OSError("File doesn't exist: %s" % args.video_path)
    if args.resolution.find('x') < 0:
        raise ValueError("Invalid resolution: %s" % args.resolution, "Should be in WxH format")

    width, height = args.resolution.split('x')
    width, height = int(width), int(height)

    image_type = IMAGE_TYPES_MAPPING['RGB']

    # Update JSON options using values from the command args
    JSON_CONFIG["assets_folder"] = args.assets
    JSON_CONFIG["charset"] = args.charset
    JSON_CONFIG["car_noplate_detect_enabled"] = (args.car_noplate_detect_enabled == "True")
    JSON_CONFIG["ienv_enabled"] = (args.ienv_enabled == "True")
    JSON_CONFIG["openvino_enabled"] = (args.openvino_enabled == "True")
    JSON_CONFIG["openvino_device"] = args.openvino_device
    JSON_CONFIG["npu_enabled"] = (args.npu_enabled == "True")
    JSON_CONFIG["klass_lpci_enabled"] = (args.klass_lpci_enabled == "True")
    JSON_CONFIG["klass_vcr_enabled"] = (args.klass_vcr_enabled == "True")
    JSON_CONFIG["klass_vmmr_enabled"] = (args.klass_vmmr_enabled == "True")
    JSON_CONFIG["klass_vbsr_enabled"] = (args.klass_vbsr_enabled == "True")
    JSON_CONFIG["license_token_file"] = args.tokenfile
    JSON_CONFIG["license_token_data"] = args.tokendata

    # Initialize the engine
    ultimateAlprSdk.UltAlprSdkEngine_init(json.dumps(JSON_CONFIG))

    capture = cv2.VideoCapture(args.video_path)

    while capture.isOpened():
        is_okay, frame = capture.read()
        if not is_okay:
            break

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        result = ultimateAlprSdk.UltAlprSdkEngine_process(
            image_type,
            frame.tobytes(),
            width,
            height,
            0, # stride
            1,
        )
        #checkResult("Process", result)

        bounding_boxes = result.json()
        print(bounding_boxes)
        break

    # DeInit
    ultimateAlprSdk.UltAlprSdkEngine_deInit()
    
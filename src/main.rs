use opencv::{
    core,
    highgui,
    imgproc,
    prelude::*,
    videoio,
};
use image::{ImageBuffer, Rgb};
use ndarray::{Array, Axis, s};
use ort::{CPUExecutionProvider, Session, SessionOutputs, inputs, GraphOptimizationLevel};
use image::imageops::FilterType;
use image::GenericImageView;
use std::error::Error;
use log::trace;

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - intersection(box1, box2)
}

const YOLOV11_CLASS_LABELS: [&str; 80] = [
    // Daftar label kelas YOLOv11
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    ort::init()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    trace!("Loading YOLOv11 model");
    let model = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file("./yolo11n.onnx")?;

    trace!("Opening camera input");
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // Kamera default
    let opened = videoio::VideoCapture::is_opened(&cap)?;
    if !opened {
        panic!("Unable to open camera!");
    }

    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    let mut frame = core::Mat::default();

    while cap.read(&mut frame)? {
        trace!("Processing frame");
        let mut img_buffer = ImageBuffer::new(frame_width as u32, frame_height as u32);
        for y in 0..frame_height {
            for x in 0..frame_width {
                let pixel = frame.at_2d::<core::Vec3b>(y, x)?;
                img_buffer.put_pixel(x as u32, y as u32, Rgb([pixel[2], pixel[1], pixel[0]]));
            }
        }

        let img = image::DynamicImage::ImageRgb8(img_buffer)
            .resize_exact(640, 640, FilterType::CatmullRom);

        let mut input = Array::zeros((1, 3, 640, 640));
        for pixel in img.pixels() {
            let x = pixel.0 as usize;
            let y = pixel.1 as usize;
            let [r, g, b] = [pixel.2[0], pixel.2[1], pixel.2[2]];
            input[[0, 0, y, x]] = (r as f32) / 255.;
            input[[0, 1, y, x]] = (g as f32) / 255.;
            input[[0, 2, y, x]] = (b as f32) / 255.;
        }

        trace!("Running YOLOv11 inference");
        let outputs: SessionOutputs = model.run(inputs!["images" => input.view()]?)?;
        let output = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();

        let mut boxes = Vec::new();
        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();

            if prob < 0.5 {
                continue;
            }

            let label = YOLOV11_CLASS_LABELS[class_id];
            let xc = row[0] / 640. * (frame_width as f32);
            let yc = row[1] / 640. * (frame_height as f32);
            let w = row[2] / 640. * (frame_width as f32);
            let h = row[3] / 640. * (frame_height as f32);

            boxes.push((
                BoundingBox {
                    x1: xc - w / 2.,
                    y1: yc - h / 2.,
                    x2: xc + w / 2.,
                    y2: yc + h / 2.
                },
                label,
                prob
            ));
        }

        boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
        let mut result = Vec::new();

        while !boxes.is_empty() {
            result.push(boxes[0]);
            boxes = boxes
                .iter()
                .filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7)
                .copied()
                .collect();
        }

        for (bbox, label, _confidence) in &result {
            let rect = core::Rect::new(
                bbox.x1 as i32,
                bbox.y1 as i32,
                (bbox.x2 - bbox.x1) as i32,
                (bbox.y2 - bbox.y1) as i32
            );

            trace!("Drawing bounding box and label for class: {}", label);
            imgproc::rectangle(
                &mut frame,
                rect,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;

            imgproc::put_text(
                &mut frame,
                label,
                core::Point::new(rect.x, rect.y - 5),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                1,
                imgproc::LINE_8,
                false,
            )?;
        }

        highgui::imshow("YOLOv11 Detection", &frame)?;
        if highgui::wait_key(1)? == 27 {  // Tombol ESC
            break;
        }
    }

    Ok(())
}

use anyhow::{anyhow, Context, Result};
use opencv::core::{Mat, Size, Vec3b, Vector};
use opencv::imgproc::INTER_LINEAR;
use opencv::prelude::*;
use opencv::{highgui, imgproc, videoio};
use tflite::model::stl::vector::VectorSlice;
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, Interpreter, InterpreterBuilder};

pub const INPUT_HEIGHT: f32 = 353.0;
pub const INPUT_WIDTH: f32 = 257.0;
pub const HEATMAP_HEIGHT: f32 = 23.0;
pub const HEATMAP_WIDTH: f32 = 17.0;
pub const NUM_KEYPOINTS: usize = 17;
pub const CONFIDENCE_THRESHOLD: f32 = 0.3; // Adjust as needed

fn main() -> Result<()> {
    // TFLite setup
    let model = FlatBufferModel::build_from_file("posenet_mobilenet.tflite")
        .context("Failed to build TFLite model")?;

    let op_resolver = BuiltinOpResolver::default();
    let builder = InterpreterBuilder::new(model, op_resolver)
        .context("Failed to create interpreter builder")?;
    let mut interpreter = builder.build().context("Failed to build interpreter")?;
    interpreter
        .allocate_tensors()
        .context("Failed to allocate tensors")?;

    // OpenCV setup
    let window = "Pose Estimation";
    highgui::named_window(window, 1).context("Failed to create window")?;
    let mut cam =
        videoio::VideoCapture::new(0, videoio::CAP_ANY).context("Failed to open default camera")?;

    if !cam.is_opened().context("Error: Camera not opened!")? {
        anyhow::bail!("Unable to open default camera!");
    }

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame).context("Failed to read frame")?;

        if frame.empty() {
            continue;
        }

        // Resize frame to 353x257 for Posenet_Mobilenet
        let mut input_frame = Mat::default();
        imgproc::resize(
            &frame,
            &mut input_frame,
            Size::new(257, 353),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )
        .context("Failed to resize frame")?;

        // Convert Mat to tflite input
        // Get the input tensor index
        let input_tensor_index = interpreter.inputs()[0];

        // Get input details
        let input_details = interpreter.get_input_details()?;
        println!("Input details: {:?}", input_details);

        // Prepare the input data
        let mut input_data = vec![0.0f32; 257 * 353 * 3];

        // Normalize and copy data
        for y in 0..353usize {
            for x in 0..257usize {
                let pixel = input_frame.at_2d::<opencv::core::Vec3b>(y as i32, x as i32)?;
                input_data[(y * 257 + x) * 3] = pixel[0] as f32 / 255.0;
                input_data[(y * 257 + x) * 3 + 1] = pixel[1] as f32 / 255.0;
                input_data[(y * 257 + x) * 3 + 2] = pixel[2] as f32 / 255.0;
            }
        }
        // Get a mutable reference to the input tensor data and copy the prepared data
        {
            let input_tensor = interpreter
                .tensor_data_mut(input_tensor_index)
                .context("Failed to get input tensor")?;
            input_tensor.copy_from_slice(&input_data);
        }

        // Run inference
        interpreter.invoke().context("Failed to run inference")?;

        // Get output tensors
        let output_indices = interpreter.outputs();
        let heatmaps: &[f32] = interpreter
            .tensor_data(output_indices[0])
            .context("Failed to get heatmaps tensor data")?;
        let short_offsets: &[f32] = interpreter
            .tensor_data(output_indices[1])
            .context("Failed to get short offsets tensor data")?;
        let mid_offsets: &[f32] = interpreter
            .tensor_data(output_indices[2])
            .context("Failed to get mid offsets tensor data")?;
        let segments: &[f32] = interpreter
            .tensor_data(output_indices[3])
            .context("Failed to get segments tensor data")?;

        // Find the top NUM_KEYPOINTS values in the heatmaps
        let top_keypoints = find_top_k(heatmaps, NUM_KEYPOINTS);

        // Get the dimensions of the display frame
        let display_width = frame.cols() as f32;
        let display_height = frame.rows() as f32;

        // Calculate scaling factors
        let scale_x = display_width / INPUT_WIDTH;
        let scale_y = display_height / INPUT_HEIGHT;

        // Process keypoints
        for keypoint_id in 0..NUM_KEYPOINTS {
            let heatmap_base = keypoint_id * (HEATMAP_HEIGHT as usize) * (HEATMAP_WIDTH as usize);
            let heatmap_slice = &heatmaps
                [heatmap_base..heatmap_base + (HEATMAP_HEIGHT as usize) * (HEATMAP_WIDTH as usize)];

            let max_index = argmax(heatmap_slice);
            let confidence = heatmap_slice[max_index];

            if confidence > CONFIDENCE_THRESHOLD {
                let heatmap_y = (max_index / HEATMAP_WIDTH as usize) as f32;
                let heatmap_x = (max_index % HEATMAP_WIDTH as usize) as f32;

                let offset_y = short_offsets[(keypoint_id
                    * 2
                    * HEATMAP_HEIGHT as usize
                    * HEATMAP_WIDTH as usize)
                    + max_index];
                let offset_x = short_offsets[(keypoint_id * 2 + 1)
                    * HEATMAP_HEIGHT as usize
                    * HEATMAP_WIDTH as usize
                    + max_index];

                let x = (heatmap_x / HEATMAP_WIDTH) * INPUT_WIDTH + offset_x;
                let y = (heatmap_y / HEATMAP_HEIGHT) * INPUT_HEIGHT + offset_y;
                let z = confidence; // Z-coordinate is represented by the confidence score

                // Scale the coordinates to the display frame
                let display_x = x * scale_x;
                let display_y = y * scale_y;

                println!(
                    "Keypoint {}: (x: {:.2}, y: {:.2}, z: {:.2})",
                    keypoint_id, display_x, display_y, z
                );

                // Draw the keypoint on the frame
                if display_x >= 0.0
                    && display_x < display_width
                    && display_y >= 0.0
                    && display_y < display_height
                {
                    let point = opencv::core::Point::new(display_x as i32, display_y as i32);
                    opencv::imgproc::circle(
                        &mut frame,
                        point,
                        3,
                        opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                        -1,
                        opencv::imgproc::LINE_8,
                        0,
                    )?;
                }
            }
        }

        // TODO: Process keypoints and draw them on the frame
        // This part depends on the exact format of your model's output
        // You'll need to interpret the keypoints and draw them on `frame`

        // Display the frame
        highgui::imshow(window, &frame).context("Failed to show image")?;

        if highgui::wait_key(1).context("Failed to wait for key")? > 0 {
            break;
        }
    }

    Ok(())
}

fn find_top_k(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed_data: Vec<(usize, f32)> =
        data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed_data.truncate(k);
    indexed_data
}

// Function to find the argmax in a slice
fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

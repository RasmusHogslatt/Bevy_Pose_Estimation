use opencv::{highgui, prelude::*, videoio, core, types, Result};
use serde::{Serialize, Deserialize};
use serde_pickle::DeOptions;
use std::fs::File;
use std::io::{Write, Read, BufReader};
use std::process::Command;
use byteorder::{ReadBytesExt, LittleEndian};
use std::thread;
use std::time::Duration;

#[derive(Serialize, Deserialize, Debug)]
struct Point3D {
    x: f32,
    y: f32,
    z: f32,
}

fn main() -> Result<()> {
    // Create named pipes before starting the process
    let _ = std::fs::remove_file("frame_pipe");
    let _ = std::fs::remove_file("points_pipe");

    std::process::Command::new("mkfifo")
        .args(&["frame_pipe", "points_pipe"])
        .status()
        .expect("Failed to create named pipes");

    // Start Python process
    let mut child = Command::new("python3")
        .arg("pestimation.py")
        .spawn()
        .expect("Failed to start Python process");

    // Sleep briefly to allow Python script to open pipes
    thread::sleep(Duration::from_secs(1));

    let mut frame_pipe = File::create("frame_pipe").expect("Failed to open frame pipe for writing");
    let mut points_pipe = File::open("points_pipe").expect("Failed to open points pipe for reading");

    let window = "3D Pose Estimation";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Unable to open default camera!");
    }

    let mut frame = core::Mat::default();
    let mut points_reader = BufReader::new(points_pipe);
    loop {
        cam.read(&mut frame)?;
        if frame.size()?.width > 0 {
            // Send frame data through pipe
            let frame_data = frame.data_bytes()?;
            frame_pipe.write_all(frame_data).expect("Failed to write frame data");

            
            // Read 3D points from pipe
            let size = points_reader.read_u32::<LittleEndian>().expect("Failed to read size");
            let mut buffer = vec![0u8; size as usize];
            points_reader.read_exact(&mut buffer).expect("Failed to read points data");
            let points: Vec<Point3D> = serde_pickle::from_slice(&buffer, DeOptions::new()).expect("Failed to deserialize points");

            if points.is_empty() {
                println!("No pose landmarks detected");
                continue;
            }
            println!("Received {} landmarks", points.len());

            // Visualize 3D points on the frame
            for point in &points {
                let x = (point.x * frame.cols() as f32) as i32;
                let y = (point.y * frame.rows() as f32) as i32;
                
                circle(&mut frame, core::Point::new(x, y), 5, core::Scalar::new(0.0, 0.0, 255.0, 0.0), -1, 8, 0)?;
            }
    
            // Draw connections
            draw_connections(&mut frame, &points)?;
            highgui::imshow(window, &frame)?;
        }
        if highgui::wait_key(10)? > 0 {
            break;
        }
    }

    // Clean up
    child.kill().expect("Failed to kill Python process");
    std::fs::remove_file("frame_pipe").expect("Failed to remove frame pipe");
    std::fs::remove_file("points_pipe").expect("Failed to remove points pipe");

    Ok(())
}

fn circle(img: &mut core::Mat, center: core::Point, radius: i32, color: core::Scalar, thickness: i32, line_type: i32, shift: i32) -> Result<()> {
    unsafe {
        opencv::imgproc::circle(
            img,
            center,
            radius,
            color,
            thickness,
            line_type,
            shift
        )?;
    }
    Ok(())
}

fn draw_connections(frame: &mut core::Mat, points: &[Point3D]) -> Result<()> {
    let connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), // Arms
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), // Torso
        (27, 29), (27, 31), (28, 30), (28, 32) // Legs
    ];

    for &(start, end) in &connections {
        if start < points.len() && end < points.len() {
            let start_point = core::Point::new(
                (points[start].x * frame.cols() as f32) as i32,
                (points[start].y * frame.rows() as f32) as i32
            );
            let end_point = core::Point::new(
                (points[end].x * frame.cols() as f32) as i32,
                (points[end].y * frame.rows() as f32) as i32
            );
            opencv::imgproc::line(frame, start_point, end_point, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, 8, 0)?;
        }
    }
    Ok(())
}
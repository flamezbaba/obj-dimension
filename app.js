const express = require("express");
const multer = require("multer");
// const tf = require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const { createCanvas, Image } = require('canvas');
const jpeg = require('jpeg-js');
const cocoSsd = require("@tensorflow-models/coco-ssd");
const fs = require("fs");
const path = require("path");

// Set up Express
const app = express();
const PORT = 3000;

// Set up Multer for file upload handling
const upload = multer({ dest: "uploads/" });

// API route to handle POST requests with image uploads
app.get("/", (req, res) => {
  res.json("it is me");
});

function imageFileToTensor(imagePath) {
  const imageBuffer = fs.readFileSync(imagePath); // Read the image file

  // Decode the JPEG image
  const decodedImage = jpeg.decode(imageBuffer, { useTArray: true });

  // Create a canvas and draw the image on it
  const { width, height, data } = decodedImage;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  
  const imageData = ctx.createImageData(width, height);
  imageData.data.set(data);
  ctx.putImageData(imageData, 0, 0);

  // Create a tensor from the image data
  const imageTensor = tf.browser.fromPixels(canvas);

  return imageTensor;
}

// Load COCO-SSD model from TensorFlow.js hosted version
async function loadModel() {
  return await tf.loadGraphModel('https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v2/model.json');
}

// Process the image and detect objects
async function detectObjects(imagePath) {
  const model =  await cocoSsd.load();

  // Convert uploaded image file to tensor using canvas and jpeg-js
  const imageTensor = imageFileToTensor(imagePath);

  // Expand dimensions to match the model input shape
  const expandedImageTensor = imageTensor.expandDims(0);

  // Perform object detection
 // Perform object detection
 const predictions = await model.detect(imageTensor);
 let objects = [];

 // Log and draw bounding boxes for each prediction
 predictions.forEach((prediction) => {
   const [x, y, width, height] = prediction.bbox;
   const area = width * height; // Calculate object area
   let sizeCategory = "";

   // Categorize based on area
   if (area < 20000) {
     sizeCategory = "small";
   } else if (area >= 20000 && area <= 100000) {
     sizeCategory = "medium";
   } else {
     sizeCategory = "large";
   }

   objects.push({
     width: width,
     height: height,
     size: sizeCategory,
     prediction: prediction.class
   });

   console.log(`Object: ${prediction.class} | 
                      Confidence: ${prediction.score.toFixed(2)} | 
                      Bounding Box: [x: ${x}, y: ${y}, width: ${width}, height: ${height}]`);
 });

 if(objects.length > 0){
   return objects[0];
 }

 return null;
}

// Example API endpoint using Express.js to handle image upload
app.post('/detect', upload.single('image'), async (req, res) => {
  try {
      // req.file contains the uploaded image information (handled by Multer)
      const imagePath = req.file.path; // Get the path of the uploaded image

      // Perform object detection
      const results = await detectObjects(imagePath);

      // Clean up uploaded file
      fs.unlinkSync(imagePath);

      res.json({ results });
  } catch (error) {
      console.error('Error detecting objects:', error);
      res.status(500).json({ error: 'Error processing image' });
  }
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
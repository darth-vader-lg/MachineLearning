using Common.Examples;
using MachineLearning.Model;
using MachineLearning.ModelZoo;
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;

namespace ObjectDetectionInference
{
   class Program
   {
      static void Main(string[] args)
      {
         // Define the data
         var modelFile = ExampleData.File(
            root: "Workspace",
            path: Path.Combine("ssd_mobilenet_v2_320x320_coco17_tpu-8", "saved_model", "saved_model.pb"),
            url: "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz");
         var imageFile = ExampleData.File(
            root: "Workspace",
            path: "banana.jpg",
            url: "https://github.com/darth-vader-lg/ML-NET/raw/056c60479304a3b5dbdf129c9bc6e853322bb090/test/data/images/banana.jpg");

         // Download the data
         Console.WriteLine($"Downloading the model...");
         var modelPath = modelFile.Get();
         Console.WriteLine($"Downloading the image...");
         var imagePath = imageFile.Get();

         // Import the model
         using var m = new ObjectDetection { ModelStorage = new ModelStorageMemory { ImportPath = modelPath } };

         // Do predictions
         var dets = m.GetPrediction(imagePath);

         // Get the boxes, draw them on bitmap and save the marked bitmap
         var boxes = dets.GetBoxes(minScore: dets.DetectionScores.Max() * 0.8);
         using (var bmp = new Bitmap(Image.FromFile(imagePath))) {
            // Draw the boxes
            foreach (var box in boxes)
               DrawBoxesOnBitmap(bmp, box);
            // Save the marked image
            var dest = Path.ChangeExtension(imagePath, null) + ".scored" + Path.GetExtension(imagePath);
            bmp.Save(dest);
            // Print the results
            Console.WriteLine($"Found {boxes.Count} objects");
            foreach (var box in boxes)
               Console.WriteLine($"{box.Name} (id:{box.Id}) {box.Score * 100f:###.#}%");
            Console.WriteLine($"The image has been saved in {dest}");
         }
      }

      /// <summary>
      /// Draw a box on a bitmap
      /// </summary>
      /// <param name="bmp">Bitmap to mark</param>
      /// <param name="box">The box to draw</param>
      private static void DrawBoxesOnBitmap(Bitmap bmp, ObjectDetection.Prediction.Box box)
      {
         using var graphic = Graphics.FromImage(bmp);
         graphic.SmoothingMode = SmoothingMode.AntiAlias;
         var rect = new Rectangle(
            (int)(box.Left * bmp.Size.Width),
            (int)(box.Top * bmp.Height),
            (int)(box.Width * bmp.Width),
            (int)(box.Height * bmp.Height));
         using var pen = new Pen(Color.Lime, Math.Max(Math.Min(rect.Width, rect.Height) / 320f, 1f));
         graphic.DrawRectangle(pen, rect);
         var fontSize = Math.Min(bmp.Size.Width, bmp.Size.Height) / 40f;
         fontSize = Math.Max(fontSize, 8f);
         fontSize = Math.Min(fontSize, rect.Height);
         using var font = new Font("Verdana", fontSize, GraphicsUnit.Pixel);
         var p = new Point(rect.Left, rect.Top);
         var text = $"{box.Name}:{(int)(box.Score * 100)}";
         var size = graphic.MeasureString(text, font);
         using var brush = new SolidBrush(Color.FromArgb(128, Color.Lime));
         graphic.FillRectangle(brush, p.X, p.Y, size.Width, size.Height);
         graphic.DrawString(text, font, Brushes.Black, p);
      }

   }
}

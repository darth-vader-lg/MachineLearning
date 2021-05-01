using Google.Protobuf;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.Json;
using Tensorflow;

namespace MachineLearning.Util
{
   /// <summary>
   /// Configurazione del modello
   /// </summary>
   internal partial class ODModelConfig
   {
      #region Fields
      /// <summary>
      /// Dizionario di conversione fra i tipi di dati Tensorflow e i dati NET
      /// </summary>
      private static readonly Dictionary<DataType, Type> tfType2Net = new()
      {
         { DataType.DtFloat, typeof(float) },
         { DataType.DtDouble, typeof(double) },
         { DataType.DtUint8, typeof(byte) },
         { DataType.DtUint16, typeof(ushort) },
         { DataType.DtUint32, typeof(uint) },
         { DataType.DtUint64, typeof(ulong) },
         { DataType.DtInt8, typeof(sbyte) },
         { DataType.DtInt16, typeof(short) },
         { DataType.DtInt32, typeof(int) },
         { DataType.DtInt64, typeof(long) },
         { DataType.DtBool, typeof(bool) },
         { DataType.DtString, typeof(string) },
      };
      #endregion
      #region Properties
      /// <summary>
      /// Formato del modello
      /// </summary>
      public ModelFormat Format { get; private set; }
      /// <summary>
      /// Dimensione dell'immagine nel modello
      /// </summary>
      public Size ImageSize { get; private set; } = new Size(640, 640);
      /// <summary>
      /// Ingressi del modello
      /// </summary>
      public ReadOnlyCollection<Tensor> Inputs { get; private set; } = new List<Tensor>().AsReadOnly();
      /// <summary>
      /// Labels
      /// </summary>
      public ReadOnlyCollection<string> Labels { get; private set; } = new List<string>().AsReadOnly();
      /// <summary>
      /// Informazioni generiche sul modello
      /// </summary>
      public JsonElement Model { get; private set; }
      /// <summary>
      /// Path del file del modello
      /// </summary>
      public string ModelFilePath { get; private set; }
      /// <summary>
      /// Tipo di modello
      /// </summary>
      public string ModelType { get; private set; }
      /// <summary>
      /// Uscite del modello
      /// </summary>
      public ReadOnlyCollection<Tensor> Outputs { get; private set; } = new List<Tensor>().AsReadOnly();
      #endregion
      #region Methods
      /// <summary>
      /// Carica la configurazione del modello
      /// </summary>
      /// <param name="path"></param>
      /// <returns></returns>
      public static ODModelConfig Load(string path)
      {
         // Configurazione
         var config = default(ODModelConfig);
         try {
            // Loop nella directory specificata o nella sottodirectory saved_model alla ricerca del modello
            foreach (var dir in new[] { path, Path.Combine(path, "saved_model") }) {
               // Ricava i path del pb
               var modelPath = dir;
               if (Directory.Exists(modelPath)) {
                  modelPath = Path.Combine(modelPath, "saved_model.pb");
                  if (!File.Exists(modelPath))
                     modelPath = Path.ChangeExtension(modelPath, ".pbtxt");
               }
               if (!File.Exists(modelPath))
                  continue;
               var configFile = Path.GetFileName(dir).ToLower() == "saved_model" ?
                  Path.Combine(Path.GetDirectoryName(dir), "saved_model.config") :
                  Path.Combine(dir, "saved_model.cfg");
               // Carica il modello
               var model = new SavedModel();
               model.MergeFrom(new CodedInputStream(File.OpenRead(modelPath)));
               // Ricerca la signature di default
               var metaGraph = model.MetaGraphs.FirstOrDefault(item => item.MetaInfoDef.Tags.Contains("serve"));
               var signature = metaGraph?.SignatureDef.FirstOrDefault(item => item.Key == "serving_default").Value;
               // Legge gli input-outputs
               if (signature != null) {
                  config = new()
                  {
                     Inputs = (from item in signature.Inputs
                               select new Tensor
                               {
                                  ColumnName = item.Value.Name,
                                  DataType = tfType2Net.TryGetValue(item.Value.Dtype, out var dataType) ? dataType : null,
                                  Dim = (from rank in item.Value.TensorShape.Dim select (int)rank.Size).ToArray(),
                                  Name = item.Key,
                               }).ToList().AsReadOnly(),
                     Outputs = (from item in signature.Outputs
                                select new Tensor
                                {
                                   ColumnName = item.Value.Name,
                                   DataType = tfType2Net.TryGetValue(item.Value.Dtype, out var dataType) ? dataType : null,
                                   Dim = (from rank in item.Value.TensorShape.Dim select (int)rank.Size).ToArray(),
                                   Name = item.Key,
                                }).ToList().AsReadOnly(),
                     Format = ModelFormat.TF2SavedModel,
                  };
               }
               // Legge le labels dal file di configurazione associato
               config.ReadInfoFromConfig(configFile);
               config.ModelFilePath = modelPath;
            }
         }
         catch (Exception) { }
         if (config == default) {
            try {
               var filePath = path;
               if (Directory.Exists(filePath))
                  filePath = Path.Combine(filePath, "saved_model.onnx");
               if (File.Exists(filePath)) {
                  using var model = new InferenceSession(filePath);
                  config = new()
                  {
                     Inputs = (from item in model.InputMetadata
                               select new Tensor
                               {
                                  ColumnName = item.Key,
                                  DataType = item.Value.ElementType,
                                  Dim = item.Value.Dimensions,
                                  Name = item.Key,
                               }).ToList().AsReadOnly(),
                     Outputs = (from item in model.OutputMetadata
                                select new Tensor
                                {
                                   ColumnName = item.Key,
                                   DataType = item.Value.ElementType,
                                   Dim = item.Value.Dimensions,
                                   Name = item.Key,
                                }).ToList().AsReadOnly(),
                     Format = ModelFormat.Onnx,
                  };
                  // Legge le labels dal file di configurazione associato
                  config.ReadInfoFromConfig(Path.ChangeExtension(filePath, ".config"));
                  config.ModelFilePath = filePath;
               }
            }
            catch (Exception exc) {
               Trace.WriteLine(exc);
            }
         }
         return config;
      }
      /// <summary>
      /// Legge le informazioni dal file di configurazione
      /// </summary>
      /// <param name="path">Path del file di configurazione</param>
      /// <returns>La lista delle labels</returns>
      private void ReadInfoFromConfig(string path)
      {
         // Legge le labels dal file di configurazione associato
         try {
            if (File.Exists(path)) {
               var jsonConfig = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(File.ReadAllText(path));
               try {
                  Labels = (from label in jsonConfig["labels"].EnumerateArray()
                            let l = new
                            {
                               Name = label.GetProperty("name").GetString(),
                               DisplayedName = label.GetProperty("display_name").GetString(),
                               Id = label.GetProperty("id").GetInt32().ToString()
                            }
                            select !string.IsNullOrEmpty(l.DisplayedName) ? l.DisplayedName : !string.IsNullOrEmpty(l.Name) ? l.Name : l.Id).ToList().AsReadOnly();
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot load the labels. {exc.Message}");
               }
               try {
                  ImageSize = new Size(jsonConfig["image_width"].GetInt32(), jsonConfig["image_height"].GetInt32());
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot load the image size. {exc.Message}");
               }
               try {
                  Model = jsonConfig["model"];
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot read the model's generic informations. {exc.Message}");
               }
               try {
                  ModelType = jsonConfig["model_type"].GetString();
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot determine the model type. {exc.Message}");
                  throw;
               }
            }
            else
               Trace.WriteLine("Warning: unable to find the configuration file associated to the model");
         }
         catch (Exception exc) {
            Trace.WriteLine($"Warning: cannot load the configuration file associated to the model. {exc.Message}");
         }
      }
      #endregion
   }

   /// <summary>
   /// Formato modello
   /// </summary>
   internal partial class ODModelConfig // ModelFormat
   {
      public enum ModelFormat
      {
         #region Definitions
         Unknown,
         TF2SavedModel,
         Onnx
         #endregion
      }
   }

   /// <summary>
   /// Tensore
   /// </summary>
   internal partial class ODModelConfig // Tensor
   {
      public class Tensor
      {
         #region Properties
         /// <summary>
         /// Nome con cui deve essere passato alle pipe
         /// </summary>
         public string ColumnName { get; set; }
         /// <summary>
         /// Tipo di dati
         /// </summary>
         public Type DataType { get; set; }
         /// <summary>
         /// Dimensioni
         /// </summary>
         public int[] Dim { get; set; }
         /// <summary>
         /// Nome del tensore
         /// </summary>
         public string Name { get; set; }
         #endregion
      }
   }
}

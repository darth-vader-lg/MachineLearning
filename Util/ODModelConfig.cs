using Google.Protobuf;
using Google.Protobuf.Collections;
using MachineLearning.ModelZoo;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
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
      /// Smart dictionary to understand the model input outputs
      /// </summary>
      private static readonly Dictionary<string, (SmartDictionary<string> Input, SmartDictionary<string> Outputs)> modelInputOutput = new()
      {
         {
            "Yolov5",
            (
               Input: new()
               {
                  { "images[1,3", "images" }
               },
               Outputs: new()
               {
                  { "[1,25200,?]", "detections" }
               }
            )
         },
         {
            "ssd",
            (
               Input: new()
               {
                  { "input", "images" }
               },
               Outputs: new()
               {
                  { "[1,25200,?]", "input_tensor:0" },
                  { "detection anchor indices[1,-1]", "detection_anchor_indices" },
                  { "detection boxes[1,-1,-1]", "detection_boxes" },
                  { "detection classes[1,-1]", "detection_classes" },
                  { "detection multiclass scores[1,-1,-1]", "detection_multiclass_scores" },
                  { "detection scores[1,-1]", "detection_scores" },
                  { "num detections[1]", "num_detections" },
                  { "raw detection boxes[1,76725,4]", "raw_detection_boxes" },
                  { "raw detection scores[1,7625,90]", "raw_detection_scores" },
               }
            )
         }
      };
      /// <summary>
      /// Smart dictionary to understand the model type
      /// </summary>
      private static readonly SmartDictionary<string> modelTypeDictionary = new()
      {
         {
            "images[1,3,640,640] "+
            "detections[1,25200,?] " +
            "grid80x80[1,3,80,80,?] " +
            "grid40x40[1,3,40,40,?] " +
            "grid20x20[1,3,20,20,?]", "Yolov5" },
         { 
            "input_tensor:0[1,-1,-1,3] " +
            "detection_anchor_indices[1,-1] " +
            "detection_boxes[1,-1,-1] " +
            "detection_classes[1,-1] " +
            "detection_multiclass_scores[1,-1,-1] " +
            "detection_scores[1,-1] " +
            "num_detections[1] " +
            "raw_detection_boxes[1,76725,4] " +
            "raw_detection_scores[1,7625,90]", "ssd"
         }
      };
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
      /// <summary>
      /// Dizionario di conversione fra i tipi di dati ONNX e i dati NET
      /// </summary>
      private static readonly Dictionary<TensorElementType, Type> onnxType2Net = new()
      {
         { TensorElementType.Bool, typeof(bool) },
         { TensorElementType.Double, typeof(double) },
         { TensorElementType.Float, typeof(float) },
         { TensorElementType.Int16, typeof(short) },
         { TensorElementType.Int32, typeof(int) },
         { TensorElementType.Int64, typeof(long) },
         { TensorElementType.Int8, typeof(sbyte) },
         { TensorElementType.String, typeof(string) },
         { TensorElementType.UInt16, typeof(ushort) },
         { TensorElementType.UInt32, typeof(uint) },
         { TensorElementType.UInt64, typeof(ulong) },
         { TensorElementType.UInt8, typeof(byte) },
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
      public Size ImageSize { get; set; } = new Size(0, 0);
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
      /// <param name="path">Path del modello</param>
      /// <returns>La sua configurazione</returns>
      public static ODModelConfig Load(string path)
      {
         // Configurazione
         var config = new ODModelConfig();
         if (Directory.GetDirectories(Path.GetDirectoryName(path)).FirstOrDefault(item => Path.GetFileName(item).ToLower() == "variables") != default)
            config.Format = ModelFormat.TF2SavedModel;
         else if (Path.GetExtension(path).ToLower() == ".pb")
            config.Format = ModelFormat.TFFrozenGraph;
         else if (Path.GetExtension(path).ToLower() == ".onnx")
            config.Format = ModelFormat.Onnx;
         else
            throw new Exception("Unknown model type");
         var configFile = path + ".config";
         // Legge i dati di configurazione
         config.ReadInfoFromConfig(configFile);
         // Completa eventuali dati mancanti
         config.ReadInfoFromModel(path);
         // Memorizza il path del modello
         config.ModelFilePath = config.Format == ModelFormat.TF2SavedModel ? Path.GetDirectoryName(path) : path;
         return config;
      }
      /// <summary>
      /// Legge le informazioni dal file di configurazione
      /// </summary>
      /// <param name="path">Path del file di configurazione</param>
      /// <returns>La lista delle labels</returns>
      private void ReadInfoFromConfig(string path)
      {
         try {
            // Verifica esistenza del file di configurazione
            if (File.Exists(path)) {
               // Legge il file di configurazione
               var jsonConfig = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(File.ReadAllText(path));
               // Ricava il formato del file
               var formatType = null as string;
               var graphType = null as string;
               Format = ModelFormat.Unknown;
               try {
                  formatType = jsonConfig["format_type"].GetString();
                  graphType = jsonConfig["graph_type"].GetString();
                  if (formatType.ToLower().Contains("tensorflow")) {
                     if (graphType.ToLower().Contains("saved_model"))
                        Format = ModelFormat.TF2SavedModel;
                     else
                        Format = ModelFormat.TFFrozenGraph;
                  }
                  else if (formatType.ToLower().Contains("onnx"))
                     Format = ModelFormat.Onnx;
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot determine the format of the model. {exc.Message}");
               }
               // Ricava il tipo di modello
               try {
                  ModelType = jsonConfig["model_type"].GetString();
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot determine the model type. {exc.Message}");
               }
               // Ricava le labels
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
               // Ricava gli inputs
               Type ToNetType(int value) =>
                  Format == ModelFormat.Onnx ?
                  (onnxType2Net.TryGetValue((TensorElementType)value, out var onnxType) ? onnxType : null) :
                  (tfType2Net.TryGetValue((DataType)value, out var tfType) ? tfType : null);
               try {
                  Inputs = (from item in jsonConfig["inputs"].EnumerateObject()
                            select new Tensor
                            {
                               ColumnName = item.Value.GetProperty("name").GetString(),
                               DataType = ToNetType(item.Value.GetProperty("type").GetInt32()),
                               Dim = (from dim in item.Value.GetProperty("shape").EnumerateArray()
                                      select dim.GetInt32()).ToArray(),
                               Name = item.Name

                            }).ToList().AsReadOnly();
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot load the inputs. {exc.Message}");
               }
               // Ricava gli outputs
               try {
                  Outputs = (from item in jsonConfig["outputs"].EnumerateObject()
                             select new Tensor
                             {
                                ColumnName = item.Value.GetProperty("name").GetString(),
                                DataType = ToNetType(item.Value.GetProperty("type").GetInt32()),
                                Dim = (from dim in item.Value.GetProperty("shape").EnumerateArray()
                                       select dim.GetInt32()).ToArray(),
                                Name = item.Name

                             }).ToList().AsReadOnly();
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot load the inputs. {exc.Message}");
               }
               // Ricava le dimensioni dell'immagine di input
               try {
                  ImageSize = new Size(jsonConfig["image_width"].GetInt32(), jsonConfig["image_height"].GetInt32());
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot load the image size. {exc.Message}");
               }
               // Memorizza altri dettagli del modello
               try {
                  Model = jsonConfig["model"];
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Warning: cannot read the model's generic informations. {exc.Message}");
               }
            }
            else
               Trace.WriteLine("Warning: unable to find the configuration file associated to the model");
         }
         catch (Exception exc) {
            Trace.WriteLine($"Warning: cannot load the configuration file associated to the model. {exc.Message}");
         }
      }
      /// <summary>
      /// Legge le eventuali informazioni mancanti direttamente dal modello
      /// </summary>
      /// <param name="path">Path del file di configurazione</param>
      /// <returns>La lista delle labels</returns>
      private void ReadInfoFromModel(string path)
      {
         try {
            // Inferenza del formato del file
            var tfModelSignature = null as SignatureDef;
            var tfGraphDef = null as GraphDef;
            var onnxModel = null as InferenceSession;
            if (Format == ModelFormat.Unknown || Inputs.Count == 0 || Outputs.Count == 0) {
               if (Format == ModelFormat.Unknown || Format == ModelFormat.TF2SavedModel) {
                  try {
                     // Tenta l'estrazione dei metagraph dal saved_model
                     var codec = FieldCodec.ForMessage(18, MetaGraphDef.Parser);
                     var metaGraphs = new RepeatedField<MetaGraphDef>();
                     var unknownFields = null as UnknownFieldSet;
                     using var stream = new CodedInputStream(File.OpenRead(path));
                     uint tag;
                     while ((tag = stream.ReadTag()) != 0) {
                        if (tag == 18)
                           metaGraphs.AddEntriesFrom(stream, codec);
                        else
                           unknownFields = UnknownFieldSet.MergeFieldFrom(unknownFields, stream);
                     }
                     // Preleva la signature
                     tfModelSignature = metaGraphs.FirstOrDefault(mg => mg.MetaInfoDef.Tags.Contains("serve"))?.SignatureDef["serving_default"];
                     if (tfModelSignature != null)
                        Format = ModelFormat.TF2SavedModel;
                  }
                  catch (Exception) { }
               }
               if (Format == ModelFormat.Unknown || Format == ModelFormat.TFFrozenGraph) {
                  try {
                     // Prova ad aprirlo come frozen_graph
                     tfGraphDef = new GraphDef();
                     tfGraphDef.MergeFrom(new CodedInputStream(File.OpenRead(path)));
                     Format = ModelFormat.TFFrozenGraph;
                  }
                  catch (Exception) { }
               }
               if (Format == ModelFormat.Unknown || Format == ModelFormat.Onnx) {
                  try {
                     // Prova ad aprirlo come onnx
                     onnxModel = new InferenceSession(path);
                     Format = ModelFormat.Onnx;
                  }
                  catch (Exception) { }
               }
            }
            // Inferenza sugli input output
            if (Inputs.Count == 0 || Outputs.Count == 0) {
               switch (Format) {
                  case ModelFormat.TF2SavedModel: {
                     static ReadOnlyCollection<Tensor> SortedTensors(IEnumerable<KeyValuePair<string, TensorInfo>> info)
                     {
                        var tensors = (from item in info
                                       select new Tensor
                                       {
                                          ColumnName = item.Value.Name,
                                          DataType = tfType2Net.TryGetValue(item.Value.Dtype, out var dataType) ? dataType : null,
                                          Dim = (from rank in item.Value.TensorShape.Dim select (int)rank.Size).ToArray(),
                                          Name = item.Key,
                                       }).ToList();
                        tensors.Sort((t1, t2) => string.Compare(t1.Name, t2.Name));
                        return tensors.AsReadOnly();
                     }

                     if (Inputs.Count == 0 && tfModelSignature != null)
                        Inputs = SortedTensors(tfModelSignature.Inputs);
                     if (Outputs.Count == 0 && tfModelSignature != null)
                        Outputs = SortedTensors(tfModelSignature.Outputs);
                     break;
                  }
                  case ModelFormat.TFFrozenGraph: {
                     // Crea un tensore da un nodo
                     static Tensor TensorFromNode(NodeDef node)
                     {
                        return new Tensor
                        {
                           ColumnName = node.Name,
                           DataType = (from dt in node.Attr
                                       where dt.Value.ValueCase == AttrValue.ValueOneofCase.Type
                                       let type = tfType2Net.TryGetValue(dt.Value.Type, out var dataType) ? dataType : null
                                       select type).FirstOrDefault(),
                           Dim = (from shape in node.Attr
                                  where shape.Value.ValueCase == AttrValue.ValueOneofCase.Shape
                                  let rank = from dim in shape.Value.Shape.Dim
                                             select (int)dim.Size
                                  select rank.ToArray()).FirstOrDefault(),
                           Name = node.Name[Math.Max(node.Name.IndexOf(':'), 0)..],
                        };
                     }
                     // Crea una lista di tensori ordinati per nome
                     static ReadOnlyCollection<Tensor> SortedTensors(IEnumerable<NodeDef> nodes)
                     {
                        var tensors = (from item in nodes select TensorFromNode(item)).ToList();
                        tensors.Sort((t1, t2) => string.Compare(t1.Name, t2.Name));
                        return tensors.AsReadOnly();
                     }
                     if (Inputs.Count == 0 && tfGraphDef != null)
                        Inputs = SortedTensors(tfGraphDef.Node.Where(item => item.Name.Contains("input_tensor")));
                     if (Outputs.Count == 0 && tfGraphDef != null)
                        Outputs = SortedTensors(tfGraphDef.Node.Where(item => item.Name.Contains("detection") && item.Op == "Identity"));
                     break;
                  }
                  case ModelFormat.Onnx: {
                     // Crea una lista di tensori ordinati per nome
                     static ReadOnlyCollection<Tensor> SortedTensors(IEnumerable<KeyValuePair<string, NodeMetadata>> nodes)
                     {
                        var tensors = (from item in nodes
                                       select new Tensor
                                       {
                                          ColumnName = item.Key,
                                          DataType = item.Value.ElementType,
                                          Dim = item.Value.Dimensions,
                                          Name = item.Key,
                                       }).ToList();
                        tensors.Sort((t1, t2) => string.Compare(t1.Name, t2.Name));
                        return tensors.AsReadOnly();
                     }
                     if (Inputs.Count == 0 && onnxModel != null)
                        Inputs = SortedTensors(onnxModel.InputMetadata);
                     if (Outputs.Count == 0 && onnxModel != null)
                        Outputs = SortedTensors(onnxModel.OutputMetadata);
                     break;
                  }
               }
            }
            // Lettura delle labels dal label_map.pbtxt o da labels.txt se collezione di labels vuota
            if (Format != ModelFormat.Unknown && Labels.Count == 0) {
               try {
                  // Legge dal label_map.pbtxt
                  var labelsFile = Path.Combine(Path.GetDirectoryName(Format == ModelFormat.TF2SavedModel ? Path.GetDirectoryName(path) : path), "label_map.pbtxt");
                  if (File.Exists(labelsFile)) {
                     var label_map = LabelMapParser.ParseFile(labelsFile);
                     var labels = new string[label_map.Items.Count];
                     foreach (var item in label_map.Items) {
                        if (item.id < 1 || item.id > labels.Length)
                           continue;
                        labels[item.id - 1] = !string.IsNullOrEmpty(item.display_name) ? item.display_name : !string.IsNullOrEmpty(item.name) ? item.name : item.id.ToString();
                     }
                     Labels = labels.ToList().AsReadOnly();
                  }
                  // Legge da labels.txt
                  else {
                     labelsFile = Path.Combine(Path.GetDirectoryName(labelsFile), "labels.txt");
                     if (File.Exists(labelsFile))
                        Labels = new List<string>(File.ReadAllLines(labelsFile)).AsReadOnly();
                  }
               }
               catch (Exception) { }
            }
            // Interpretazione del tipo di modello
            if (string.IsNullOrEmpty(ModelType)) {
               var sb = new StringBuilder();
               foreach (var io in new[] { Inputs, Outputs }) {
                  foreach (var item in io) {
                     sb.Append($"{(sb.Length > 0 ? " " : "")}{item.Name}");
                     if (item.Dim != null) {
                        var sbDim = new StringBuilder();
                        sbDim.Append('[');
                        foreach (var d in item.Dim)
                           sbDim.Append($"{(sbDim.Length > 1 ? "," : "")}{d}");
                        sbDim.Append(']');
                        sb.Append(sbDim);
                     }
                  }
               }
               ModelType = modelTypeDictionary.Similar[sb.ToString()];
            }
            // Interpretazione della dimensione dell'immagine
            if (ImageSize.Width < 1 || ImageSize.Height < 1) {
               try {
                  ImageSize = ModelType switch
                  {
                     "Yolov5" => new Size(Inputs[0].Dim[2], Inputs[0].Dim[3]),
                     _ => new Size(Inputs[0].Dim[1], Inputs[0].Dim[2]),
                  };
                  if (ImageSize.Width < 1 || ImageSize.Height < 1)
                     throw new Exception("Cannot infer image size from the input tensor");
               }
               catch (Exception exc) {
                  Trace.WriteLine($"Error: unknown image size. {exc.Message}");
                  throw;
               }
            }
         }
         catch (Exception exc) {
            Trace.WriteLine($"Warning: cannot load the configuration file associated to the model. {exc.Message}");
         }
      }
      #endregion
   }

   /// <summary>
   /// Parser labels
   /// </summary>
   internal partial class ODModelConfig // LabelMapItem, LabelMapItems, LabelMapParser
   {
      /// <summary>
      /// Item di un file di labels pbtxt
      /// </summary>
      internal class LabelMapItem
      {
         #region Properties
         /// <summary>
         /// Nome da visualizzare
         /// </summary>
         [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "<Minuscolo necessario alla deserializzazione>")]
         public string display_name { get; set; }
         /// <summary>
         /// Identificatore
         /// </summary>
         [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "<Minuscolo necessario alla deserializzazione>")]
         public int id { get; set; }
         /// <summary>
         /// Nome della label
         /// </summary>
         [SuppressMessage("Style", "IDE1006:Stili di denominazione", Justification = "<Minuscolo necessario alla deserializzazione>")]
         public string name { get; set; }
         #endregion
      }

      /// <summary>
      /// Elenco di items di un file di labels pbtxt
      /// </summary>
      internal class LabelMapItems
      {
         #region Properties
         /// <summary>
         /// Elementi
         /// </summary>
         public List<LabelMapItem> Items { get; set; }
         #endregion
      }

      /// <summary>
      /// Parser di un file di labels
      /// </summary>
      internal class LabelMapParser
      {
         #region Methods
         /// <summary>
         /// Parserizza il file e restituisce gli elementi
         /// </summary>
         /// <param name="filePath">Path del file</param>
         /// <returns>L'elenco di itens</returns>
         internal static LabelMapItems ParseFile(string filePath)
         {
            string line;
            var newText = "{\"Items\":[";
            using (var reader = new System.IO.StreamReader(filePath)) {
               while ((line = reader.ReadLine()) != null) {
                  var newline = string.Empty;
                  if (line.Contains("{")) {
                     newline = line.Replace("item", "").Trim();
                     newText += newline;
                  }
                  else if (line.Contains("}")) {
                     newText = newText.Remove(newText.Length - 1);
                     newText += line;
                     newText += ",";
                  }
                  else {
                     newline = line.Replace(":", "\":").Trim();
                     newline = "\"" + newline;
                     newline += ",";
                     newText += newline;
                  }
               }
               newText = newText.Remove(newText.Length - 1);
               newText += "]}";
               reader.Close();
            }
            var items = JsonSerializer.Deserialize<LabelMapItems>(newText);
            return items;
         }
         #endregion
      }
   }

   /// <summary>
   /// Formato modello
   /// </summary>
   internal partial class ODModelConfig // ModelFormat
   {
      internal enum ModelFormat
      {
         #region Definitions
         Unknown,
         TF2SavedModel,
         TFFrozenGraph,
         Onnx
         #endregion
      }
   }

   /// <summary>
   /// Tensore
   /// </summary>
   internal partial class ODModelConfig // Tensor
   {
      internal class Tensor
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

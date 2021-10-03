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
using System.Reflection;
using System.Text;
using System.Text.Json;
using Tensorflow;

namespace MachineLearning.Model
{
   /// <summary>
   /// Class to infer automatically the type of a model
   /// </summary>
   internal partial class ModelConfig
   {
      #region Fields
      /// <summary>
      /// Smart dictionary to understand the model type
      /// </summary>
      private static readonly SmartDictionary<(ModelInfo Model, string[] Labels)> modelConfigsDictionary = new();
      /// <summary>
      /// Dictionary of tensors per model
      /// </summary>
      private static readonly Dictionary<string, Dictionary<ColumnTypes, Tensor>> tensorsDictionaries = new();
      /// <summary>
      /// Dictionary of tensors
      /// </summary>
      private Dictionary<ColumnTypes, Tensor> tensorsDictionary = new();
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
      /// Fingerprint of the model
      /// </summary>
      public string Fingerprint { get; private set; }
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
      /// Model category
      /// </summary>
      public string ModelCategory { get; private set; }
      /// <summary>
      /// Path del file del modello
      /// </summary>
      public string ModelFilePath { get; private set; }
      /// <summary>
      /// Tipo di modello
      /// </summary>
      public string ModelType { get; private set; }
      /// <summary>
      /// Offset that must apply to the pixels of the images
      /// </summary>
      public float OffsetImage { get; private set; } = 0f;
      /// <summary>
      /// Uscite del modello
      /// </summary>
      public ReadOnlyCollection<Tensor> Outputs { get; private set; } = new List<Tensor>().AsReadOnly();
      /// <summary>
      /// Scale factor that must apply to the image pixel values
      /// </summary>
      public float ScaleImage { get; private set; } = 1f;
      #endregion
      #region Methods
      /// <summary>
      /// Return the name of the requested column type
      /// </summary>
      /// <param name="type">Type of the column</param>
      /// <returns>The column name</returns>
      public string GetColumnName(ColumnTypes type) => tensorsDictionary[type].ColumnName;
      /// <summary>
      /// Return the tensor of the requested column type
      /// </summary>
      /// <param name="type">Type of the column</param>
      /// <returns>The tensor</returns>
      public Tensor GetTensor(ColumnTypes type) => tensorsDictionary[type];
      /// <summary>
      /// Load the model configuration from path
      /// </summary>
      /// <param name="path">Path of the model</param>
      /// <param name="modelConfig">Optional different configuration file</param>
      /// <returns>The configuration</returns>
      public static ModelConfig Load(string path, string modelConfig = default)
      {
         // Models dictionary
         var modelConfigsDictionary = default(SmartDictionary<(ModelInfo Model, string[] Labels)>);
         lock (ModelConfig.modelConfigsDictionary) {
            if (modelConfig != null) {
               if (!File.Exists(modelConfig))
                  throw new FileNotFoundException("The model configuration file doesn't exist", modelConfig);
               var cfg = JsonSerializer.Deserialize<ModelConfigs>(File.ReadAllText(modelConfig));
               modelConfigsDictionary = new SmartDictionary<(ModelInfo Model, string[] Labels)>(from m in cfg.Models
                                                                                                let kv = (Key: string.Join(' ', m.Fingerprint), Value: (Model: m, Labels: cfg.Labels[m.Labels]))
                                                                                                select new KeyValuePair<string, (ModelInfo Model, string[] Labels)>(kv.Key, kv.Value));
            }
            else if (ModelConfig.modelConfigsDictionary.Count == 0) {
               var cfg = JsonSerializer.Deserialize<ModelConfigs>(File.ReadAllText(Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "ModelConfigs.json")));
               foreach (var item in from m in cfg.Models
                                    let kv = (Key: string.Join(' ', m.Fingerprint), Value: (Model: m, Labels: cfg.Labels[m.Labels]))
                                    select new KeyValuePair<string, (ModelInfo Model, string[] Labels)>(kv.Key, kv.Value))
                  ModelConfig.modelConfigsDictionary.Add(item);
               modelConfigsDictionary = ModelConfig.modelConfigsDictionary;
            }
            else
               modelConfigsDictionary = ModelConfig.modelConfigsDictionary;
         }
         // Configuration
         var config = new ModelConfig();
         if (Directory.GetDirectories(Path.GetDirectoryName(path)).FirstOrDefault(item => Path.GetFileName(item).ToLower() == "variables") != default)
            config.Format = ModelFormat.TF2SavedModel;
         else if (Path.GetExtension(path).ToLower() == ".pb")
            config.Format = ModelFormat.TFFrozenGraph;
         else if (Path.GetExtension(path).ToLower() == ".onnx")
            config.Format = ModelFormat.Onnx;
         else
            throw new Exception("Unknown model type");
         var configFile = path + ".config";
         // Read config data
         config.ReadInfoFromConfig(configFile);
         // Complete possible missing data
         config.ReadInfoFromModel(path, modelConfigsDictionary);
         // Fill the dictionary of model's column types
         lock (tensorsDictionaries) {
            if (!tensorsDictionaries.TryGetValue(config.Fingerprint, out config.tensorsDictionary)) {
               config.tensorsDictionary = new();
               var modelTensors = new SmartDictionary<Dictionary<ColumnTypes, string>>(from m in modelConfigsDictionary
                                                                                       select new KeyValuePair<string, Dictionary<ColumnTypes, string>>(m.Value.Model.Type, m.Value.Model.ColumnType));
               // Fill the dictionaries to convert from column type to tensor info
               var tensorLookup = modelTensors.Similar[config.ModelType + config.ModelCategory];
               if (config.Inputs.Count > 1) {
                  var inputTensorInfo = new SmartDictionary<Tensor>(from t in config.Inputs select new KeyValuePair<string, Tensor>(t.ToString(), t));
                  if (tensorLookup.ContainsKey(ColumnTypes.Input))
                     config.tensorsDictionary[ColumnTypes.Input] = inputTensorInfo.Similar[tensorLookup[ColumnTypes.Input]];
               }
               else
                  config.tensorsDictionary[ColumnTypes.Input] = config.Inputs[0];
               var outputTensorInfo = config.Outputs.Count > 1 ? new SmartDictionary<Tensor>(from t in config.Outputs select new KeyValuePair<string, Tensor>(t.ToString(), t)) : null;
               foreach (var item in from n in Enum.GetNames<ColumnTypes>() where n != ColumnTypes.Input.ToString() select n) {
                  var ct = Enum.Parse<ColumnTypes>(item);
                  if (!tensorLookup.ContainsKey(ct))
                     continue;
                  config.tensorsDictionary[ct] = outputTensorInfo?.Similar[tensorLookup[ct]] ?? config.Outputs[0];
               }
               tensorsDictionaries[config.Fingerprint] = config.tensorsDictionary;
            }
         }
         // Store model path
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
                  var labelsDict =
                     jsonConfig["labels"].EnumerateArray()
                     .Select(item => (Name: item.GetProperty("name").GetString(), DisplayedName: item.GetProperty("display_name").GetString(), Id: item.GetProperty("id").GetInt32()))
                     .Select(item => (item.Id, Name: !string.IsNullOrEmpty(item.DisplayedName) ? item.DisplayedName : !string.IsNullOrEmpty(item.Name) ? item.Name : item.Id.ToString()))
                     .ToDictionary(item => item.Id);
                  var maxId = labelsDict.Max(item => item.Key);
                  var labels =
                     Enumerable.Range(0, maxId + 1)
                     .Select(id => labelsDict.TryGetValue(id, out var item) ? item.Name : id.ToString())
                     .ToList();
                  Labels = labels.AsReadOnly();
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
                  Trace.WriteLine($"Warning: cannot load the outputs. {exc.Message}");
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
      /// Read the missing information directly from the model
      /// </summary>
      /// <param name="path">Model path</param>
      /// <param name="modelConfigsDictionary">Known models dictionary</param>
      /// <returns>La lista delle labels</returns>
      private void ReadInfoFromModel(string path, SmartDictionary<(ModelInfo Model, string[] Labels)> modelConfigsDictionary)
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
                  catch (Exception exc) {
                     Trace.WriteLine(exc);
                  }
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
                     var inputNames = new[] { "input_tensor", "input" };
                     if (Inputs.Count == 0 && tfGraphDef != null)
                        Inputs = SortedTensors(tfGraphDef.Node.Where(item => inputNames.Any(text => item.Name.Contains(text)) && item.Op == "Placeholder"));
                     var outputNames = new[] { "detection", "output" };
                     if (Outputs.Count == 0 && tfGraphDef != null)
                        Outputs = SortedTensors(tfGraphDef.Node.Where(item => outputNames.Any(text => item.Name.Contains(text)) && item.Op == "Identity"));
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
            // Fingerprint of the model
            var sb = new StringBuilder();
            foreach (var io in new[] { Inputs, Outputs }) {
               foreach (var item in io)
                  sb.Append($"{(sb.Length > 0 ? " " : "")}{item}");
            }
            Fingerprint = sb.ToString();
            // Model type inferring
            var preferredSize = default(Size);
            var standardLabels = default(string[]);
            if (string.IsNullOrEmpty(ModelType)) {
               var info = modelConfigsDictionary.Similar[Fingerprint];
               ModelType = info.Model.Type;
               ModelCategory = info.Model.Category;
               ScaleImage = info.Model.ScaleImage;
               OffsetImage = info.Model.OffsetImage;
               preferredSize = info.Model.ImageSize?.Length == 2 ? new Size(info.Model.ImageSize[0], info.Model.ImageSize[1]) : default;
               standardLabels = info.Labels;
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
                     else if (standardLabels != null && standardLabels.Length > 0)
                        Labels = standardLabels.ToList().AsReadOnly();
                  }
               }
               catch (Exception) { }
            }
            // Interpretazione della dimensione dell'immagine
            if (ImageSize.Width < 1 || ImageSize.Height < 1) {
               try {
                  if (ModelType.ToLower().Contains("yolo"))
                     ImageSize = new Size(Inputs[0].Dim[2], Inputs[0].Dim[3]);
                  else
                     ImageSize = Inputs[0].Dim?.Length >= 3 ? new Size(Inputs[0].Dim[1], Inputs[0].Dim[2]) : default;
                  if (ImageSize.Width < 1 || ImageSize.Height < 1) {
                     if (preferredSize.Width > 0 && preferredSize.Height > 0)
                        ImageSize = preferredSize;
                  }
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
   /// Column types
   /// </summary>
   internal partial class ModelConfig // ColumnTypes
   {
      internal enum ColumnTypes
      {
         #region Definitions
         Boxes,
         Classes,
         Input,
         Scores,
         #endregion
      }
   }

   /// <summary>
   /// Labels parser
   /// </summary>
   internal partial class ModelConfig // LabelMapItem, LabelMapItems, LabelMapParser
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
            using (var reader = new StreamReader(filePath)) {
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
   /// Model format
   /// </summary>
   internal partial class ModelConfig // ModelFormat
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
   /// Model info
   /// </summary>
   internal partial class ModelConfig // ModelInfo, ModelConfigs
   {
      [Serializable]
      internal class ModelInfo
      {
         #region Properties
         /// <summary>
         /// Type of the model
         /// </summary>
         public string Type { get; set; } = "Unknown";
         /// <summary>
         /// Category of the model
         /// </summary>
         public string Category { get; set; } = "Unknown";
         /// <summary>
         /// Fingerprint of the model based on it's tensors
         /// </summary>
         public string[] Fingerprint { get; set; } = Array.Empty<string>();
         /// <summary>
         /// Default image size
         /// </summary>
         public int[] ImageSize { get; set; } = default;
         /// <summary>
         /// Dictionary of Column type -> tensor
         /// </summary>
         public Dictionary<ColumnTypes, string> ColumnType { get; set; } = new();
         /// <summary>
         /// Associated labels
         /// </summary>
         public string Labels { get; set; } = null;
         /// <summary>
         /// Offset that must apply to the pixels of the images
         /// </summary>
         public float OffsetImage { get; set; } = 0f;
         /// <summary>
         /// Scale factor that must apply to the image pixel values
         /// </summary>
         public float ScaleImage { get; set; } = 1f;
         #endregion
      }
      [Serializable]
      internal class ModelConfigs
      {
         #region Properties
         /// <summary>
         /// Set of know models
         /// </summary>
         public ModelInfo[] Models { get; set; } = Array.Empty<ModelInfo>();
         /// <summary>
         /// Labels dictionary
         /// </summary>
         public Dictionary<string, string[]> Labels { get; set; } = new();
         #endregion
      }
   }

   /// <summary>
   /// Tensor
   /// </summary>
   internal partial class ModelConfig // Tensor
   {
      internal class Tensor
      {
         #region Properties
         /// <summary>
         /// Name on the DataViewSchema
         /// </summary>
         public string ColumnName { get; set; }
         /// <summary>
         /// Data type
         /// </summary>
         public Type DataType { get; set; }
         /// <summary>
         /// Dimensions
         /// </summary>
         public int[] Dim { get; set; }
         /// <summary>
         /// Name of the tensor
         /// </summary>
         public string Name { get; set; }
         /// <summary>
         /// String representation
         /// </summary>
         /// <returns></returns>
         public override string ToString()
         {
            var sb = new StringBuilder();
            sb.Append($"{DataType.Name} {Name}({ColumnName})");
            if (Dim != null) {
               sb.Append('[');
               for (var i = 0; i < Dim.Length; i++)
                  sb.Append($"{(i > 0 ? "," : "")}{(Dim[i] > -1 ? Dim[i] : "?")}");
               sb.Append(']');
            }
            return sb.ToString();
         }
         #endregion
      }
   }
}

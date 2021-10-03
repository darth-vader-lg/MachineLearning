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
   internal partial class ModelConfig : IDisposable
   {
      #region Fields
      /// <summary>
      /// Disposed object
      /// </summary>
      private bool disposedValue;
      /// <summary>
      /// Pixel colors interleaving required
      /// </summary>
      private bool? interleavePixelColors;
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
      /// Offset that must apply to the pixels of the images
      /// </summary>
      private float? offsetImage;
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
      /// <summary>
      /// Scale factor that must apply to the image pixel values
      /// </summary>
      private float? scaleImage;
      #endregion
      #region Properties
      /// <summary>
      /// Fingerprint of the model
      /// </summary>
      public string Fingerprint { get; private set; }
      /// <summary>
      /// Model format
      /// </summary>
      public ModelFormat Format { get; private set; }
      /// <summary>
      /// Model image's input dimension
      /// </summary>
      public Size ImageSize { get; set; } = new Size(0, 0);
      /// <summary>
      /// Model inputs
      /// </summary>
      public ReadOnlyCollection<Tensor> Inputs { get; private set; } = new List<Tensor>().AsReadOnly();
      /// <summary>
      /// Pixel colors interleaving required
      /// </summary>
      public bool InterleavePixelColors => interleavePixelColors ?? false;
      /// <summary>
      /// Labels
      /// </summary>
      public ReadOnlyCollection<string> Labels { get; private set; } = new List<string>().AsReadOnly();
      /// <summary>
      /// Generic information
      /// </summary>
      public JsonElement Model { get; private set; }
      /// <summary>
      /// Model category
      /// </summary>
      public string ModelCategory { get; private set; }
      /// <summary>
      /// Final path of the model
      /// </summary>
      public string ModelFilePath { get; private set; }
      /// <summary>
      /// Model type
      /// </summary>
      public string ModelType { get; private set; }
      /// <summary>
      /// Offset that must apply to the pixels of the images
      /// </summary>
      public float OffsetImage => offsetImage ?? 0f;
      /// <summary>
      /// Original model file path
      /// </summary>
      public string OriginalFilePath { get; private set; }
      /// <summary>
      /// Model outputs
      /// </summary>
      public ReadOnlyCollection<Tensor> Outputs { get; private set; } = new List<Tensor>().AsReadOnly();
      /// <summary>
      /// Scale factor that must apply to the image pixel values
      /// </summary>
      public float ScaleImage => scaleImage ?? 1f;
      /// <summary>
      /// The scorer name for the model
      /// </summary>
      public string Scorer { get; set; } = default;
      #endregion
      #region Methods
      /// <summary>
      /// Finalizer
      /// </summary>
      ~ModelConfig() =>
         Dispose(disposing: false);
      /// <summary>
      /// Convert a model from PyTorch to Onnx
      /// </summary>
      /// <param name="src">PyTorch file name</param>
      /// <returns>The path of the converted model</returns>
      private static string ConvertFromPyTorchToOnnx(string src)
      {
         var dst = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".onnx");
         ODModelBuilderTF.Converter.Convert(src, dst, ODModelBuilderTF.Converter.Formats.PyTorch, ODModelBuilderTF.Converter.Formats.Onnx);
         File.SetLastWriteTimeUtc(dst, File.GetLastWriteTimeUtc(src));
         return dst;
      }
      /// <summary>
      /// Dispose
      /// </summary>
      public void Dispose()
      {
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Dispose
      /// </summary>
      /// <param name="disposing">tru if called by the program</param>
      protected virtual void Dispose(bool disposing)
      {
         if (disposedValue)
            return;
         if (disposing) {
            // Free managed resources
         }
         try {
            if (!string.IsNullOrEmpty(ModelFilePath) && string.Compare(ModelFilePath, OriginalFilePath, true) != 0)
               File.Delete(ModelFilePath);
         }
         catch { }
         tensorsDictionary = null;
         disposedValue = true;
      }
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
      /// <param name="modelType">Optional model type specification</param>
      /// <param name="modelCategory">Optional model category specification</param>
      /// <param name="modelConfig">Optional different configuration file</param>
      /// <remarks>
      /// modelType and modelCategory could be similar and not strictly exact spelt when doing automatic model kind inference.
      /// They must be exact just if specified and if exists a model configuration.
      /// So it's better to use these parameters only in automatic model kind inference.
      /// </remarks>
      /// <returns>The configuration</returns>
      public static ModelConfig Load(string path, string modelType = default, string modelCategory = default, string modelConfig = default)
      {
         // Models dictionary
         var modelConfigsDictionary = default(SmartDictionary<(ModelInfo Model, string[] Labels)>);
         lock (ModelConfig.modelConfigsDictionary) {
            if (modelConfig != null) {
               if (!File.Exists(modelConfig))
                  throw new FileNotFoundException("The model configuration file doesn't exist", modelConfig);
               var cfg = JsonSerializer.Deserialize<ModelConfigs>(File.ReadAllText(modelConfig));
               modelConfigsDictionary = new SmartDictionary<(ModelInfo Model, string[] Labels)>(from m in cfg.Models
                                                                                                let fingerprint = string.Join(' ', new[] { m.Type, m.Category }.Concat(m.Fingerprint))
                                                                                                let kv = (Key: fingerprint, Value: (Model: m, Labels: cfg.Labels[m.Labels]))
                                                                                                select new KeyValuePair<string, (ModelInfo Model, string[] Labels)>(kv.Key, kv.Value));
            }
            else if (ModelConfig.modelConfigsDictionary.Count == 0) {
               var cfg = JsonSerializer.Deserialize<ModelConfigs>(File.ReadAllText(Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "ModelConfigs.json")));
               foreach (var item in from m in cfg.Models
                                    let fingerprint = string.Join(' ', new[] { m.Type, m.Category }.Concat(m.Fingerprint))
                                    let kv = (Key: fingerprint, Value: (Model: m, Labels: cfg.Labels[m.Labels]))
                                    select new KeyValuePair<string, (ModelInfo Model, string[] Labels)>(kv.Key, kv.Value))
                  ModelConfig.modelConfigsDictionary.Add(item);
               modelConfigsDictionary = ModelConfig.modelConfigsDictionary;
            }
            else
               modelConfigsDictionary = ModelConfig.modelConfigsDictionary;
         }
         // Configuration
         var config = new ModelConfig() { OriginalFilePath = Path.GetFullPath(path) };
         if (Directory.GetDirectories(Path.GetDirectoryName(path)).FirstOrDefault(item => Path.GetFileName(item).ToLower() == "variables") != default)
            config.Format = ModelFormat.TF2SavedModel;
         else if (Path.GetExtension(path).ToLower() == ".pb")
            config.Format = ModelFormat.TFFrozenGraph;
         else if (Path.GetExtension(path).ToLower() == ".onnx")
            config.Format = ModelFormat.Onnx;
         else if (Path.GetExtension(path).ToLower() == ".pt") {
            try {
               path = ConvertFromPyTorchToOnnx(path);
               config.Format = ModelFormat.Onnx;
            }
            catch (Exception exc) {
               throw new InvalidOperationException("Cannot convert from PyTorch", exc);
            }
         }
         var configFile = path + ".config";
         // Read config data
         config.ReadInfoFromConfig(configFile, modelType, modelCategory);
         // Complete possible missing data
         config.ReadInfoFromModel(path, modelType, modelCategory, modelConfigsDictionary);
         // Check data
         if (string.IsNullOrEmpty(config.ModelType))
            throw new InvalidOperationException("Unknown model type");
         if (config.Format == ModelFormat.Unknown)
            throw new InvalidOperationException("Unknown model format");
         if (config.Inputs.Count == 0)
            throw new InvalidOperationException("Unknown input tensors");
         if (config.Outputs.Count == 0)
            throw new InvalidOperationException("Unknown output tensors");
         // Fill the dictionary of model's column types
         lock (tensorsDictionaries) {
            if (!tensorsDictionaries.TryGetValue(config.Fingerprint, out config.tensorsDictionary)) {
               config.tensorsDictionary = new();
               var modelTensors = new SmartDictionary<Dictionary<ColumnTypes, string>>(from modelInfo in modelConfigsDictionary
                                                                                       let m = modelInfo.Value.Model
                                                                                       let fingerprint = string.Join(' ', new[] { m.Type, m.Category }.Concat(m.Fingerprint))
                                                                                       let columnType = m.ColumnType
                                                                                       select new KeyValuePair<string, Dictionary<ColumnTypes, string>>(fingerprint, columnType));
               // Fill the dictionaries to convert from column type to tensor info
               var tensorLookup = modelTensors.Similar[config.Fingerprint];
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
      /// <param name="modelType">Model type specification</param>
      /// <param name="modelCategory">Model category specification</param>
      /// <returns>La lista delle labels</returns>
      private void ReadInfoFromConfig(string path, string modelType, string modelCategory)
      {
         // Check the existence of the configuration file
         if (!File.Exists(path))
            return;
         // Read the configuration
         var jsonConfig = default(Dictionary<string, JsonElement>);
         try {
            jsonConfig = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(File.ReadAllText(path));
         }
         catch (Exception exc) {
            throw new InvalidDataException($"Invalid configration file {path}", exc);
         }
         // get the file format
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
            Debug.WriteLine($"Warning: unknown model format in the configuration file. {exc.Message}");
         }
         // Get the model type
         try {
            ModelType = jsonConfig["model_type"].GetString();
         }
         catch (Exception exc) {
            if (string.IsNullOrEmpty(modelType))
               Debug.WriteLine($"Warning: unknown model type in the configuration file. {exc.Message}");
         }
         if (!string.IsNullOrEmpty(modelType)) {
            if (!string.IsNullOrWhiteSpace(ModelType) && modelType != ModelType)
               throw new InvalidOperationException($"The specified model type {modelType} doesn't match the configuration's model type {ModelType}");
            ModelType = modelType;
         }
         // Get the model category
         try {
            ModelCategory = jsonConfig["model_category"].GetString();
         }
         catch (Exception exc) {
            if (string.IsNullOrEmpty(modelCategory))
               Debug.WriteLine($"Warning: unknown model category in the configuration file. {exc.Message}");
         }
         if (!string.IsNullOrEmpty(modelCategory)) {
            if (!string.IsNullOrWhiteSpace(ModelCategory) && modelCategory != ModelCategory)
               throw new InvalidOperationException($"The specified model category {modelCategory} doesn't match the configuration's model category {ModelCategory}");
            ModelCategory = modelCategory;
         }
         // Get the labels
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
            Debug.WriteLine($"Warning: cannot load the labels. {exc.Message}");
         }
         // Get the inputs
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
            Debug.WriteLine($"Warning: cannot load the inputs. {exc.Message}");
         }
         // Get the outputs
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
            Debug.WriteLine($"Warning: cannot load the outputs. {exc.Message}");
         }
         // Get the input image's dimensions
         try {
            ImageSize = new Size(jsonConfig["image_width"].GetInt32(), jsonConfig["image_height"].GetInt32());
         }
         catch (Exception exc) {
            Debug.WriteLine($"Warning: cannot load the image size. {exc.Message}");
         }
         // Get the offset for pixels of the image
         try {
            offsetImage = jsonConfig["offset_image"].GetSingle();
         }
         catch (Exception exc) {
            Debug.WriteLine($"Warning: cannot load the offset image. {exc.Message}");
         }
         // Get the scale factor for pixels of the image
         try {
            scaleImage = jsonConfig["scale_image"].GetSingle();
         }
         catch (Exception exc) {
            Debug.WriteLine($"Warning: cannot load the scale image. {exc.Message}");
         }
         // Get the interleave requirement for the pixels colors
         try {
            interleavePixelColors = jsonConfig["interleave"].GetBoolean();
         }
         catch (Exception exc) {
            Debug.WriteLine($"Warning: cannot load the interlave image pixels enable. {exc.Message}");
         }
         // Get scorer type
         try {
            Scorer = jsonConfig["scorer"].GetString();
         }
         catch (Exception exc) {
            Debug.WriteLine($"Warning: cannot load scorer type. {exc.Message}");
         }
         // Store other model details
         try {
            Model = jsonConfig["model"];
         }
         catch (Exception exc) {
            Debug.WriteLine($"Warning: cannot read the model's generic informations. {exc.Message}");
         }
      }
      /// <summary>
      /// Read the missing information directly from the model
      /// </summary>
      /// <param name="path">Model path</param>
      /// <param name="modelType">Model type specification</param>
      /// <param name="modelCategory">Model category specification</param>
      /// <param name="modelConfigsDictionary">Known models dictionary</param>
      /// <returns>La lista delle labels</returns>
      private void ReadInfoFromModel(string path, string modelType, string modelCategory, SmartDictionary<(ModelInfo Model, string[] Labels)> modelConfigsDictionary)
      {
         // File format inference
         var tfModelSignature = null as SignatureDef;
         var tfGraphDef = null as GraphDef;
         var onnxModel = null as InferenceSession;
         if (Format == ModelFormat.Unknown || Inputs.Count == 0 || Outputs.Count == 0) {
            if (Format == ModelFormat.Unknown || Format == ModelFormat.TF2SavedModel) {
               try {
                  // Try to extract the metagraph from the saved_model
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
                  // Try to open as frozen graph
                  tfGraphDef = new GraphDef();
                  tfGraphDef.MergeFrom(new CodedInputStream(File.OpenRead(path)));
                  Format = ModelFormat.TFFrozenGraph;
               }
               catch (Exception) { }
            }
            if (Format == ModelFormat.Unknown || Format == ModelFormat.Onnx) {
               try {
                  // Try to open as onnx
                  onnxModel = new InferenceSession(path);
                  Format = ModelFormat.Onnx;
               }
               catch (Exception) { }
            }
            if (Format == ModelFormat.Unknown)
               throw new InvalidOperationException("Unknown model format");
         }
         // Input / outputs inference
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
         if (modelType != default)
            sb.Append(modelType);
         if (modelCategory != default)
            sb.Append(modelCategory);
         foreach (var io in new[] { Inputs, Outputs }) {
            foreach (var item in io)
               sb.Append($"{(sb.Length > 0 ? " " : "")}{item}");
         }
         Fingerprint = sb.ToString();
         // Missing info inferring
         var info = default((ModelInfo Model, string[] Labels));
         (ModelInfo Model, string[] Labels) GetInfo()
         {
            if (info == default)
               info = modelConfigsDictionary.Similar[Fingerprint];
            return info;
         }
         if (string.IsNullOrEmpty(ModelType))
            ModelType = GetInfo().Model.Type;
         if (string.IsNullOrEmpty(ModelCategory))
            ModelCategory = GetInfo().Model.Category;
         if (scaleImage == null)
            scaleImage = GetInfo().Model.ScaleImage;
         if (offsetImage == null)
            offsetImage = GetInfo().Model.OffsetImage;
         if (interleavePixelColors == null)
            interleavePixelColors = GetInfo().Model.InterleavePixelColors;
         if (Scorer == null)
            Scorer = GetInfo().Model.Scorer;
         // Fill the labels
         if (Format != ModelFormat.Unknown && Labels.Count == 0) {
            try {
               // Read from label_map.pbtxt
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
               // Read from labels.txt
               else {
                  labelsFile = Path.Combine(Path.GetDirectoryName(labelsFile), "labels.txt");
                  if (File.Exists(labelsFile))
                     Labels = new List<string>(File.ReadAllLines(labelsFile)).AsReadOnly();
                  else {
                     var standardLabels = GetInfo().Labels;
                     if (standardLabels != null && standardLabels.Length > 0)
                        Labels = standardLabels.ToList().AsReadOnly();
                  }
               }
            }
            catch (Exception exc)
            {
               Debug.WriteLine($"Error reading the labels. {exc.Message}");
            }
         }
         // Image dimension inference
         if (ImageSize.Width < 1 || ImageSize.Height < 1) {
            var infoSize = GetInfo().Model.ImageSize;
            var preferredSize = infoSize?.Length == 2 ? new Size(infoSize[0], infoSize[1]) : default;
            if (preferredSize.Width > 0 && preferredSize.Height > 0)
               ImageSize = preferredSize;
         }
         if (ImageSize.Width < 1 || ImageSize.Height < 1) {
            try {
               if (ModelType.ToLower().Contains("yolo"))
                  ImageSize = new Size(Inputs[0].Dim[2], Inputs[0].Dim[3]);
               else
                  ImageSize = Inputs[0].Dim?.Length >= 3 ? new Size(Inputs[0].Dim[1], Inputs[0].Dim[2]) : default;
            }
            catch (Exception exc) {
               Debug.WriteLine($"Error reading the image size from input tensor. {exc.Message}");
            }
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
         /// Pixel colors interleaving required
         /// </summary>
         public bool InterleavePixelColors { get; set; } = false;
         /// <summary>
         /// Offset that must apply to the pixels of the images
         /// </summary>
         public float OffsetImage { get; set; } = 0f;
         /// <summary>
         /// Scale factor that must apply to the image pixel values
         /// </summary>
         public float ScaleImage { get; set; } = 1f;
         /// <summary>
         /// The scorer name for the model
         /// </summary>
         public string Scorer { get; set; } = default;
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

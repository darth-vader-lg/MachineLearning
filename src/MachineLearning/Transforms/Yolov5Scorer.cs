using MachineLearning.Transforms;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;

[assembly: LoadableClass(YoloV5ScorerTransformer.Summary, typeof(IDataTransform), typeof(YoloV5ScorerTransformer), typeof(YoloV5ScorerEstimator.Options), typeof(SignatureDataTransform),
    YoloV5ScorerTransformer.UserName)]

[assembly: LoadableClass(YoloV5ScorerTransformer.Summary, typeof(IDataTransform), typeof(YoloV5ScorerTransformer), null, typeof(SignatureLoadDataTransform),
    YoloV5ScorerTransformer.UserName, YoloV5ScorerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(YoloV5ScorerTransformer), null, typeof(SignatureLoadModel),
    YoloV5ScorerTransformer.UserName, YoloV5ScorerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(YoloV5ScorerTransformer), null, typeof(SignatureLoadRowMapper),
    YoloV5ScorerTransformer.UserName, YoloV5ScorerTransformer.LoaderSignature)]

namespace MachineLearning.Transforms
{
   /// <summary>
   /// Transformer for the output of the YoloV5 CNN
   /// </summary>
   public sealed partial class YoloV5ScorerTransformer : RowToRowTransformerBase
   {
      #region Fields
      /// <summary>
      /// Name of the input column (the output of the Yolo model)
      /// </summary>
      private readonly string inputScoresColumnName;
      /// <summary>
      /// The height of the model image
      /// </summary>
      private readonly int imageHeight;
      /// <summary>
      /// The width of the model image
      /// </summary>
      private readonly int imageWidth;
      /// <summary>
      /// Set of the labels associate with the model
      /// </summary>
      internal readonly string[] labels;
      /// <summary>
      /// Signature for the loader
      /// </summary>
      internal const string LoaderSignature = "YoloV5Scorer";
      /// <summary>
      /// Summary for the loader
      /// </summary>
      internal const string Summary = "Score the data of a Yolo V5 object detection CNN";
      /// <summary>
      /// Minimum score confidence for each category
      /// </summary>
      private readonly float minPerCategoryConfidence;
      /// <summary>
      /// Minimum score confidence for each single cell
      /// </summary>
      private readonly float minScoreConfidence;
      /// <summary>
      /// Output column with detection boxes in format left, top, right, bottom from 0 to 1
      /// </summary>
      private readonly string outputBoxesColumnName;
      /// <summary>
      /// Output column with detection classes
      /// </summary>
      private readonly string outputClassesColumnName;
      /// <summary>
      /// Output column with detection labels
      /// </summary>
      private readonly string outputLabelsColumnName;
      /// <summary>
      /// Output column with detection scores
      /// </summary>
      private readonly string outputScoresColumnName;
      /// <summary>
      /// User name for the loader
      /// </summary>
      internal const string UserName = "YoloV5Scorer";
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="outputClassesColumnName">Output column with detection classes</param>
      /// <param name="outputScoresColumnName">Output column with detection scores</param>
      /// <param name="outputBoxesColumnName">Output column with detection boxes in format left, top, right, bottom from 0 to 1</param>
      /// <param name="outputLabelsColumnName">Output column with detection labels</param>
      /// <param name="inputScoresColumnName">Name of the input column (the output of the Yolo model)</param>
      /// <param name="minScoreConfidence">Minimum score confidence for each single cell</param>
      /// <param name="minPerCategoryConfidence">Minimum score confidence for each category</param>
      /// <param name="imageWidth">The width of the model image</param>
      /// <param name="imageHeight">The height of the model image</param>
      /// <param name="labels">Set of the labels associate with the model</param>
      internal YoloV5ScorerTransformer(
         IHostEnvironment env,
         string outputClassesColumnName = YoloV5ScorerEstimator.Options.DefaultClassesColumnName,
         string outputScoresColumnName = YoloV5ScorerEstimator.Options.DefaultScoresColumnName,
         string outputBoxesColumnName = YoloV5ScorerEstimator.Options.DefaultBoxesColumnName,
         string outputLabelsColumnName = YoloV5ScorerEstimator.Options.DefaultLabelsColumnName,
         string inputScoresColumnName = YoloV5ScorerEstimator.Options.DefaultInputColumnName,
         float minScoreConfidence = YoloV5ScorerEstimator.Options.DefaultMinScoreConfidence,
         float minPerCategoryConfidence = YoloV5ScorerEstimator.Options.DefaultMinPerCategoryConfidence,
         int imageWidth = YoloV5ScorerEstimator.Options.DefaultImageWidth,
         int imageHeight = YoloV5ScorerEstimator.Options.DefaultImageHeight,
         IEnumerable<string> labels = null)
         : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(YoloV5ScorerTransformer)))
      {
         Host.CheckValue(outputClassesColumnName, nameof(outputClassesColumnName));
         Host.CheckValue(outputScoresColumnName, nameof(outputScoresColumnName));
         Host.CheckValue(outputBoxesColumnName, nameof(outputBoxesColumnName));
         Host.CheckValue(outputLabelsColumnName, nameof(outputLabelsColumnName));
         Host.CheckParam(imageWidth > 0, nameof(imageWidth));
         Host.CheckParam(imageHeight > 0, nameof(imageHeight));
         this.outputClassesColumnName = outputClassesColumnName;
         this.outputScoresColumnName = outputScoresColumnName;
         this.outputBoxesColumnName = outputBoxesColumnName;
         this.outputLabelsColumnName = outputLabelsColumnName;
         this.inputScoresColumnName = !string.IsNullOrWhiteSpace(inputScoresColumnName) ? inputScoresColumnName : outputScoresColumnName;
         this.minPerCategoryConfidence = minPerCategoryConfidence;
         this.minScoreConfidence = minScoreConfidence;
         this.imageWidth = imageWidth;
         this.imageHeight = imageHeight;
         this.labels = labels?.ToArray();
      }
      /// <summary>
      /// Factory method for SignatureLoadModel.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="ctx">Model load context</param>
      internal YoloV5ScorerTransformer(IHostEnvironment env, ModelLoadContext ctx)
         : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(YoloV5ScorerTransformer)))
      {
         Host.CheckValue(ctx, nameof(ctx));
         ctx.CheckAtModel(GetVersionInfo());
         inputScoresColumnName = ctx.LoadNonEmptyString();
         outputClassesColumnName = ctx.LoadNonEmptyString();
         outputScoresColumnName = ctx.LoadNonEmptyString();
         outputBoxesColumnName = ctx.LoadNonEmptyString();
         outputLabelsColumnName = ctx.LoadNonEmptyString();
         minPerCategoryConfidence = float.Parse(ctx.LoadNonEmptyString(), CultureInfo.InvariantCulture);
         minScoreConfidence = float.Parse(ctx.LoadNonEmptyString(), CultureInfo.InvariantCulture);
         imageWidth = int.Parse(ctx.LoadNonEmptyString(), CultureInfo.InvariantCulture);
         imageHeight = int.Parse(ctx.LoadNonEmptyString(), CultureInfo.InvariantCulture);
         var numLabels = ctx.Reader.ReadInt32();
         if (numLabels > 0) {
            var labels = new string[numLabels];
            for (var i = 0; i < numLabels; i++)
               labels[i] = ctx.LoadNonEmptyString();
            this.labels = labels;
         }
      }
      /// <summary>
      /// Factory method for SignatureDataTransform.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="options">Options</param>
      /// <param name="input">Input data</param>
      internal static IDataTransform Create(IHostEnvironment env, YoloV5ScorerEstimator.Options options, IDataView input)
      {
         Contracts.CheckValue(env, nameof(env));
         env.CheckValue(options, nameof(options));
         env.CheckValue(input, nameof(input));
         env.CheckValue(options.InputScoresColumnName, nameof(options.InputScoresColumnName));
         env.CheckValue(options.OutputClassesColumnName, nameof(options.OutputClassesColumnName));
         env.CheckValue(options.OutputScoresColumnName, nameof(options.OutputScoresColumnName));
         env.CheckValue(options.OutputBoxesColumnName, nameof(options.OutputBoxesColumnName));
         env.CheckValue(options.OutputLabelsColumnName, nameof(options.OutputLabelsColumnName));
         env.CheckParam(options.ImageWidth > 0, nameof(options.ImageWidth));
         env.CheckParam(options.ImageHeight > 0, nameof(options.ImageHeight));
         return new YoloV5ScorerEstimator(env, options).Fit(input).MakeDataTransform(input);
      }
      /// <summary>
      /// Factory method for SignatureLoadDataTransform.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="ctx">Load context</param>
      /// <param name="input">Input data</param>
      private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
          => new YoloV5ScorerTransformer(env, ctx).MakeDataTransform(input);
      /// <summary>
      /// Factory method for SignatureLoadRowMapper.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="ctx">Load context</param>
      /// <param name="inputSchema">Input schema</param>
      private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
          => new YoloV5ScorerTransformer(env, ctx).MakeRowMapper(inputSchema);
      /// <summary>
      /// Get the annotations for the classes column, if labels are defined
      /// </summary>
      /// <returns>The annotations or null</returns>
      internal DataViewSchema.Annotations GetClassesAnnotations()
      {
         // Build the annotations containing the slots (label names)
         var annotations = default(DataViewSchema.Annotations);
         if (labels != null) {
            var builder = new DataViewSchema.Annotations.Builder();
            builder.AddSlotNames(labels.Length, (ref VBuffer<ReadOnlyMemory<char>> value) =>
            {
               value = new VBuffer<ReadOnlyMemory<char>>(labels.Length, labels.Select(label => label.AsMemory()).ToArray());
            });
            annotations = builder.ToAnnotations();
         }
         return annotations;
      }
      /// <summary>
      /// Get the info about the version
      /// </summary>
      /// <returns></returns>
      private static VersionInfo GetVersionInfo()
      {
         return new VersionInfo(
            modelSignature: "YOLV5SCR", // Yolo V5 scorer transformer
            verWrittenCur: 0x00010001, // Initial
            verReadableCur: 0x00010001,
            verWeCanReadBack: 0x00010001,
            loaderSignature: LoaderSignature,
            loaderAssemblyName: typeof(YoloV5ScorerTransformer).Assembly.FullName);
      }
      /// <summary>
      /// Create the mapper
      /// </summary>
      /// <param name="schema">Input schema</param>
      /// <returns>The mapper</returns>
      private protected override IRowMapper MakeRowMapper(DataViewSchema schema) =>
         new Mapper(this, schema);
      /// <summary>
      /// Save the model
      /// </summary>
      /// <param name="ctx">Save context</param>
      private protected override void SaveModel(ModelSaveContext ctx)
      {
         Host.AssertValue(ctx);
         ctx.CheckAtModel();
         ctx.SetVersionInfo(GetVersionInfo());
         Host.AssertNonWhiteSpace(inputScoresColumnName);
         Host.AssertNonWhiteSpace(outputClassesColumnName);
         Host.AssertNonWhiteSpace(outputScoresColumnName);
         Host.AssertNonWhiteSpace(outputBoxesColumnName);
         Host.AssertNonWhiteSpace(outputLabelsColumnName);
         Host.Assert(imageWidth > 0);
         Host.Assert(imageHeight > 0);
         ctx.SaveNonEmptyString(inputScoresColumnName);
         ctx.SaveNonEmptyString(outputClassesColumnName);
         ctx.SaveNonEmptyString(outputScoresColumnName);
         ctx.SaveNonEmptyString(outputBoxesColumnName);
         ctx.SaveNonEmptyString(outputLabelsColumnName);
         ctx.SaveString(minPerCategoryConfidence.ToString(CultureInfo.InvariantCulture));
         ctx.SaveString(minScoreConfidence.ToString(CultureInfo.InvariantCulture));
         ctx.SaveString(imageWidth.ToString(CultureInfo.InvariantCulture));
         ctx.SaveString(imageHeight.ToString(CultureInfo.InvariantCulture));
         ctx.Writer.Write(labels?.Length ?? 0);
         if (labels != null) {
            foreach (var label in labels)
               ctx.SaveNonEmptyString(label);
         }
      }
      #endregion
   }

   partial class YoloV5ScorerTransformer // Mapper
   {
      /// <summary>
      /// The mapper for <see cref="YoloV5ScorerTransformer"/>
      /// </summary>
      private sealed partial class Mapper : MapperBase
      {
         #region Fields
         /// <summary>
         /// Cache for outputs
         /// </summary>
         private readonly Cache cache;
         /// <summary>
         /// Index of the input column in the schema
         /// </summary>
         private readonly int inputColumnIndex;
         /// <summary>
         /// The associated transformer
         /// </summary>
         private readonly YoloV5ScorerTransformer transformer;
         #endregion
         #region Methods
         /// <summary>
         /// Constructor
         /// </summary>
         /// <param name="transformer">The associated transformer</param>
         /// <param name="inputSchema"></param>
         public Mapper(YoloV5ScorerTransformer transformer, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(transformer, nameof(transformer)).Host.Register(nameof(Mapper)), inputSchema, transformer)
         {
            Host.CheckValue(transformer, nameof(transformer));
            this.transformer = transformer;
            // Check presence of input columns.
            if (!inputSchema.TryGetColumnIndex(this.transformer.inputScoresColumnName, out inputColumnIndex))
               throw Host.ExceptSchemaMismatch(nameof(inputSchema), "source", this.transformer.inputScoresColumnName);
            // Create the cache
            cache = new Cache(this);
         }
         /// <summary>
         /// Get the dependencies
         /// </summary>
         /// <param name="activeOutput">Active output indicator</param>
         /// <returns>The dependency functions</returns>
         private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput) =>
            col => Enumerable.Range(0, 4).Any(i => activeOutput(i)) && inputColumnIndex == col;
         /// <summary>
         /// Get the output columns
         /// </summary>
         /// <returns>The array of output columns</returns>
         protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
         {
            return new DataViewSchema.DetachedColumn[]
            {
               new DataViewSchema.DetachedColumn(transformer.outputClassesColumnName, new VectorDataViewType(NumberDataViewType.Int32), transformer.GetClassesAnnotations()),
               new DataViewSchema.DetachedColumn(transformer.outputScoresColumnName, new VectorDataViewType(NumberDataViewType.Single)),
               new DataViewSchema.DetachedColumn(transformer.outputBoxesColumnName, new VectorDataViewType(NumberDataViewType.Single, 0, 4)),
               new DataViewSchema.DetachedColumn(transformer.outputLabelsColumnName, new VectorDataViewType(TextDataViewType.Instance)),
            };
         }
         /// <summary>
         /// Create the getter for the results
         /// </summary>
         /// <param name="input">Input row</param>
         /// <param name="iinfo">Index of the output</param>
         /// <param name="activeOutput">Active output indicator</param>
         /// <param name="disposer">Disposer of data</param>
         /// <returns>The delegate of the getter</returns>
         protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
         {
            Host.AssertValue(input);
            disposer = null;
            return iinfo switch
            {
               0 => (ValueGetter<VBuffer<int>>)((ref VBuffer<int> value) => value = cache.Get(input).Classes),
               1 => (ValueGetter<VBuffer<float>>)((ref VBuffer<float> value) => value = cache.Get(input).Scores),
               2 => (ValueGetter<VBuffer<float>>)((ref VBuffer<float> value) => value = cache.Get(input).Boxes),
               3 => (ValueGetter<VBuffer<ReadOnlyMemory<char>>>)((ref VBuffer<ReadOnlyMemory<char>> value) => value = cache.Get(input).Labels),
               _ => null,
            };
         }
         /// <summary>
         /// Save the model
         /// </summary>
         /// <param name="ctx">Save context</param>
         private protected override void SaveModel(ModelSaveContext ctx)
             => transformer.SaveModel(ctx);
         #endregion
      }
   }

   partial class YoloV5ScorerTransformer // Cached values for the mapper
   {
      partial class Mapper
      {
         /// <summary>
         /// The cache for the mapper <see cref="YoloV5ScorerTransformer.Mapper"/>
         /// </summary>
         internal class Cache
         {
            #region Fields
            /// <summary>
            /// The transformer
            /// </summary>
            private readonly Mapper mapper;
            #endregion
            #region Properties
            /// <summary>
            /// Cached boxes
            /// </summary>
            internal VBuffer<float> Boxes { get; private set; }
            /// <summary>
            /// Cached classes
            /// </summary>
            internal VBuffer<int> Classes { get; private set; }
            /// <summary>
            /// Cached labels
            /// </summary>
            internal VBuffer<ReadOnlyMemory<char>> Labels { get; private set; }
            /// <summary>
            /// Row position of the cache
            /// </summary>
            internal long? Position { get; private set; }
            /// <summary>
            /// Cached scores
            /// </summary>
            internal VBuffer<float> Scores { get; private set; }
            #endregion
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="mapper">The associated mapper</param>
            internal Cache(Mapper mapper) =>
               this.mapper = mapper;
            /// <summary>
            /// Get the cache for the current input row
            /// </summary>
            /// <param name="input">Input row</param>
            /// <returns>The cache</returns>
            internal Cache Get(DataViewRow input)
            {
               // Cache the results if needed
               if (input.Position == Position)
                  return this;
               lock (this) {
                  Position = input.Position;
                  var yoloOutput = default(VBuffer<float>);
                  input.GetGetter<VBuffer<float>>(input.Schema[mapper.inputColumnIndex]).Invoke(ref yoloOutput);
                  var values = yoloOutput.GetValues().ToArray();
                  // Loop over detection blocks
                  var characteristics = values.Length / ((80 * 80 + 40 * 40 + 20 * 20) * 3);
                  var numClasses = characteristics - 5;
                  var results = new List<(RectangleF Box, float Class, float Score, bool Valid)>();
                  for (int i = 0; i < values.Length; i += characteristics) {
                     // Filter boxes
                     var objConf = values[i + 4];
                     if (objConf <= mapper.transformer.minScoreConfidence)
                        continue;
                     // Real classes score and search for the one with max score
                     var maxConf = -float.MaxValue;
                     var maxClass = -1;
                     for (var j = 0; j < numClasses; j++) {
                        var score = values[i + 5 + j] * objConf;
                        if (score > maxConf) {
                           maxConf = score;
                           maxClass = j;
                        }
                     }
                     // Check if maximum score is above minimum threshold
                     if (maxConf < mapper.transformer.minPerCategoryConfidence)
                        continue;
                     // Add result
                     var xc = values[i + 1] / mapper.transformer.imageWidth;   // X centre
                     var yc = values[i + 0] / mapper.transformer.imageHeight;  // Y centre
                     var w = values[i + 3] / mapper.transformer.imageWidth;    // Width
                     var h = values[i + 2] / mapper.transformer.imageHeight;   // Height
                     results.Add((Box: new RectangleF(xc - w / 2, yc - h / 2, w, h), Class: maxClass + 1, Score: maxConf, Valid: true));
                  }
                  // NMS filter
                  var nmsOverlapRatio = 0.45f; //@@@
                  if (nmsOverlapRatio < 1f) {
                     for (var i = 0; i < results.Count; i++) {
                        var item = results[i];
                        if (!item.Valid)
                           continue;
                        for (var j = 0; j < results.Count; j++) {
                           var current = results[j];
                           if (current == item || !current.Valid)
                              continue;
                           var intersection = RectangleF.Intersect(item.Box, current.Box);
                           var intArea = intersection.Width * intersection.Height;
                           var unionArea = item.Box.Width * item.Box.Height + current.Box.Width * current.Box.Height - intArea;
                           var overlap = intArea / unionArea;
                           if (overlap > nmsOverlapRatio) {
                              if (item.Score > current.Score) {
                                 current.Valid = false;
                                 results[j] = current;
                              }
                           }
                        }
                     }
                  }
                  // Store the ready results
                  var validResults = results.Where(item => item.Valid).ToArray();
                  var boxesData = validResults.SelectMany(item => new[] { item.Box.Left, item.Box.Top, item.Box.Right, item.Box.Bottom }).Select(v => v).ToArray();
                  Boxes = new VBuffer<float>(boxesData.Length, boxesData);
                  var classesData = validResults.Select(item => (int)item.Class).ToArray();
                  Classes = new VBuffer<int>(classesData.Length, classesData);
                  var scoresData = validResults.Select(item => item.Score).ToArray();
                  Scores = new VBuffer<float>(scoresData.Length, scoresData);
                  var labelsData = default(ReadOnlyMemory<char>[]);
                  if (mapper.transformer.labels != null)
                     labelsData = classesData.Select(item => mapper.transformer.labels[item - 1].AsMemory()).ToArray();
                  else
                     labelsData = classesData.Select(item => item.ToString().AsMemory()).ToArray();
                  Labels = new VBuffer<ReadOnlyMemory<char>>(labelsData.Length, labelsData);
               }
               return this;
            }
         }
      }
   }

   /// <summary>
   /// Estimator for the <see cref="YoloV5ScorerTransformer"/>.
   /// </summary>
   public sealed partial class YoloV5ScorerEstimator : TrivialEstimator<YoloV5ScorerTransformer>
   {
      #region Fields
      /// <summary>
      /// The options
      /// </summary>
      private readonly Options options = new();
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="env">Host environment</param>
      /// <param name="options">Scorer options</param>
      internal YoloV5ScorerEstimator(IHostEnvironment env, Options options)
         : base(
            Contracts.CheckRef(env, nameof(env)).Register(nameof(YoloV5ScorerEstimator)),
            new YoloV5ScorerTransformer(
               env,
               options.OutputClassesColumnName,
               options.OutputScoresColumnName,
               options.OutputBoxesColumnName,
               options.OutputLabelsColumnName,
               options.InputScoresColumnName,
               options.MinScoreConfidence,
               options.MinPerCategoryConfidence,
               options.ImageWidth,
               options.ImageHeight,
               options.Labels))
      {
         Host.CheckNonEmpty(options.OutputClassesColumnName, nameof(options.OutputClassesColumnName));
         Host.CheckNonEmpty(options.OutputScoresColumnName, nameof(options.OutputScoresColumnName));
         Host.CheckNonEmpty(options.OutputBoxesColumnName, nameof(options.OutputBoxesColumnName));
         Host.CheckNonEmpty(options.OutputLabelsColumnName, nameof(options.OutputLabelsColumnName));
         Host.Check(options.ImageWidth > 0, "Image width must be > 0");
         Host.Check(options.ImageHeight > 0, "Image height must be > 0");
         this.options.ImageHeight = options.ImageHeight;
         this.options.ImageWidth = options.ImageWidth;
         this.options.InputScoresColumnName = !string.IsNullOrWhiteSpace(options.InputScoresColumnName) ? options.InputScoresColumnName : options.OutputScoresColumnName;
         this.options.Labels = options.Labels;
         this.options.MinPerCategoryConfidence = options.MinPerCategoryConfidence;
         this.options.MinScoreConfidence = options.MinScoreConfidence;
         this.options.OutputBoxesColumnName = options.OutputBoxesColumnName;
         this.options.OutputClassesColumnName = options.OutputClassesColumnName;
         this.options.OutputLabelsColumnName = options.OutputLabelsColumnName;
         this.options.OutputScoresColumnName = options.OutputScoresColumnName;
      }
      /// <summary>
      /// Get the sketch of the output schema 
      /// </summary>
      /// <param name="inputSchema">Input schema</param>
      /// <returns>The output</returns>
      public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
      {
         Host.CheckValue(inputSchema, nameof(inputSchema));
         var result = inputSchema.ToDictionary(x => x.Name);
         if (!inputSchema.TryFindColumn(options.InputScoresColumnName, out var col))
            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", options.InputScoresColumnName);
         if (col.Kind != SchemaShape.Column.VectorKind.Vector || col.ItemType != NumberDataViewType.Single)
            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", options.InputScoresColumnName, "Vector of Single", col.GetTypeString());
         var annotation = default(SchemaShape);
         var classesAnnotations = Transformer.GetClassesAnnotations();
         if (classesAnnotations != null) {
            annotation = new SchemaShape(new[]
            {
               new SchemaShape.Column(
                  classesAnnotations.Schema[0].Name,
                  SchemaShape.Column.VectorKind.Vector,
                  TextDataViewType.Instance,
                  false)
            });
         }
         result[options.OutputBoxesColumnName] = new SchemaShape.Column(
             options.OutputBoxesColumnName,
             SchemaShape.Column.VectorKind.Vector,
             NumberDataViewType.Single,
             false);
         result[options.OutputClassesColumnName] = new SchemaShape.Column(
             options.OutputClassesColumnName,
             SchemaShape.Column.VectorKind.Vector,
             NumberDataViewType.Int32,
             false,
             annotation);
         result[options.OutputLabelsColumnName] = new SchemaShape.Column(
             options.OutputLabelsColumnName,
             SchemaShape.Column.VectorKind.Vector,
             TextDataViewType.Instance,
             false);
         result[options.OutputScoresColumnName] = new SchemaShape.Column(
             options.OutputScoresColumnName,
             SchemaShape.Column.VectorKind.Vector,
             NumberDataViewType.Single,
             false);
         return new SchemaShape(result.Values);
      }
      #endregion
   }

   partial class YoloV5ScorerEstimator // Options
   {
      /// <summary>
      /// The options for the <see cref="YoloV5ScorerTransformer"/>.
      /// </summary>
      public sealed class Options : TransformInputBase
      {
         #region Fields
         /// <summary>
         /// Default name for the detection classes column
         /// </summary>
         public const string DefaultClassesColumnName = "detection_classes";
         /// <summary>
         /// Default name for the detection scores column
         /// </summary>
         public const string DefaultScoresColumnName = "detection_scores";
         /// <summary>
         /// Default name for the detection boxes column
         /// </summary>
         public const string DefaultBoxesColumnName = "detection_boxes";
         /// <summary>
         /// Default name for the detection labels column
         /// </summary>
         public const string DefaultLabelsColumnName = "detection_labels";
         /// <summary>
         /// Default height of the model image
         /// </summary>
         public const int DefaultImageHeight = 640;
         /// <summary>
         /// Default width of the model image
         /// </summary>
         public const int DefaultImageWidth = 640;
         /// <summary>
         /// Default name for the input column
         /// </summary>
         public const string DefaultInputColumnName = "detection";
         /// <summary>
         /// Default minimum score confidence for each single cell
         /// </summary>
         public const float DefaultMinScoreConfidence = 0.2f;
         /// <summary>
         /// Default minimum score confidence for each category
         /// </summary>
         public const float DefaultMinPerCategoryConfidence = 0.25f;
         /// <summary>
         /// The name of the column containing the inputs for the scorer (the output of the Yolo model).
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input column (the output of the Yolo model)", Name = "InputScores", ShortName = "is", SortOrder = 1)]
         public string InputScoresColumnName = DefaultInputColumnName;
         /// <summary>
         /// The name of the output column containing the detection classes.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the output column with detection classes", Name = "OutputClasses", ShortName = "oc", SortOrder = 2)]
         public string OutputClassesColumnName = DefaultClassesColumnName;
         /// <summary>
         /// The name of the output column containing the detection scores.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the output column with detection scores", Name = "OutputScores", ShortName = "os", SortOrder = 3)]
         public string OutputScoresColumnName = DefaultScoresColumnName;
         /// <summary>
         /// The name of the output column containing the detection boxes.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the output column with detection boxes", Name = "OutputBoxes", ShortName = "ob", SortOrder = 4)]
         public string OutputBoxesColumnName = DefaultBoxesColumnName;
         /// <summary>
         /// The name of the output column containing the detection labels.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the output column with detection labels", Name = "OutputLabels", ShortName = "ol", SortOrder = 5)]
         public string OutputLabelsColumnName = DefaultLabelsColumnName;
         /// <summary>
         /// Minimum score confidence for each single cell
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum score for each single cell", Name = "Confidence", ShortName = "c", SortOrder = 6)]
         public float MinScoreConfidence = DefaultMinScoreConfidence;
         /// <summary>
         /// Minimum score confidence for each category
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum score for each category", Name = "CategoryConfidence", ShortName = "cc", SortOrder = 7)]
         public float MinPerCategoryConfidence = DefaultMinPerCategoryConfidence;
         /// <summary>
         /// The width of the model image
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Image width of the model", Name = "Width", ShortName = "w", SortOrder = 8)]
         public int ImageWidth = DefaultImageWidth;
         /// <summary>
         /// The height of the model image
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Image height of the model", Name = "Height", ShortName = "h", SortOrder = 9)]
         public int ImageHeight = DefaultImageHeight;
         /// <summary>
         /// Set of labels
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "List of labels associated to the model", Name = "Labels", ShortName = "l", SortOrder = 10)]
         public IEnumerable<string> Labels = null;
         #endregion
      }
   }
}

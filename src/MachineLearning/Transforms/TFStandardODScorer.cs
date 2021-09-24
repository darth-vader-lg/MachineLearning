using MachineLearning.Transforms;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

[assembly: LoadableClass(TFStandardODScorerTransformer.Summary, typeof(IDataTransform), typeof(TFStandardODScorerTransformer), typeof(TFStandardODScorerEstimator.Options), typeof(SignatureDataTransform),
    TFStandardODScorerTransformer.UserName)]

[assembly: LoadableClass(TFStandardODScorerTransformer.Summary, typeof(IDataTransform), typeof(TFStandardODScorerTransformer), null, typeof(SignatureLoadDataTransform),
    TFStandardODScorerTransformer.UserName, TFStandardODScorerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(TFStandardODScorerTransformer), null, typeof(SignatureLoadModel),
    TFStandardODScorerTransformer.UserName, TFStandardODScorerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TFStandardODScorerTransformer), null, typeof(SignatureLoadRowMapper),
    TFStandardODScorerTransformer.UserName, TFStandardODScorerTransformer.LoaderSignature)]

namespace MachineLearning.Transforms
{
   /// <summary>
   /// Transformer for the output of the TensorFlow object detection NN
   /// </summary>
   public sealed partial class TFStandardODScorerTransformer : RowToRowTransformerBase
   {
      #region Fields
      /// <summary>
      /// Input column with detection boxes in format left, top, right, bottom from 0 to 1
      /// </summary>
      private readonly string inputBoxesColumnName;
      /// <summary>
      /// Input column with detection classes
      /// </summary>
      private readonly string inputClassesColumnName;
      /// <summary>
      /// Input column with detection scores
      /// </summary>
      private readonly string inputScoresColumnName;
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
      /// Set of the labels associate with the model
      /// </summary>
      internal readonly string[] labels;
      /// <summary>
      /// Signature for the loader
      /// </summary>
      internal const string LoaderSignature = "TFStandardODScorer";
      /// <summary>
      /// Summary for the loader
      /// </summary>
      internal const string Summary = "Score the data of a standard TensorFlow object detection NN";
      /// <summary>
      /// Minimum score of detections
      /// </summary>
      private readonly float minScore;
      /// <summary>
      /// User name for the loader
      /// </summary>
      internal const string UserName = "TFStandardODScorer";
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
      /// <param name="inputClassesColumnName">Input column with detection classes</param>
      /// <param name="inputScoresColumnName">Input column with detection scores</param>
      /// <param name="inputBoxesColumnName">Input column with detection boxes in format left, top, right, bottom from 0 to 1</param>
      /// <param name="minScore">Minimum score for detection</param>
      /// <param name="labels">Set of the labels associate with the model</param>
      internal TFStandardODScorerTransformer(
         IHostEnvironment env,
         string outputClassesColumnName = TFStandardODScorerEstimator.Options.DefaultClassesColumnName,
         string outputScoresColumnName = TFStandardODScorerEstimator.Options.DefaultScoresColumnName,
         string outputBoxesColumnName = TFStandardODScorerEstimator.Options.DefaultBoxesColumnName,
         string outputLabelsColumnName = TFStandardODScorerEstimator.Options.DefaultLabelsColumnName,
         string inputClassesColumnName = TFStandardODScorerEstimator.Options.DefaultClassesColumnName,
         string inputScoresColumnName = TFStandardODScorerEstimator.Options.DefaultScoresColumnName,
         string inputBoxesColumnName = TFStandardODScorerEstimator.Options.DefaultBoxesColumnName,
         float minScore = TFStandardODScorerEstimator.Options.DefaultMinScore,
         IEnumerable<string> labels = null)
         : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TFStandardODScorerTransformer)))
      {
         Host.CheckValue(outputClassesColumnName, nameof(outputClassesColumnName));
         Host.CheckValue(outputScoresColumnName, nameof(outputScoresColumnName));
         Host.CheckValue(outputBoxesColumnName, nameof(outputBoxesColumnName));
         Host.CheckValue(outputLabelsColumnName, nameof(outputLabelsColumnName));
         this.outputClassesColumnName = outputClassesColumnName;
         this.outputScoresColumnName = outputScoresColumnName;
         this.outputBoxesColumnName = outputBoxesColumnName;
         this.outputLabelsColumnName = outputLabelsColumnName;
         this.inputClassesColumnName = !string.IsNullOrWhiteSpace(inputClassesColumnName) ? inputClassesColumnName : outputClassesColumnName;
         this.inputScoresColumnName = !string.IsNullOrWhiteSpace(inputScoresColumnName) ? inputScoresColumnName : outputScoresColumnName;
         this.inputBoxesColumnName = !string.IsNullOrWhiteSpace(inputBoxesColumnName) ? inputBoxesColumnName : outputBoxesColumnName;
         this.minScore = minScore;
         this.labels = labels?.ToArray();
      }
      /// <summary>
      /// Factory method for SignatureLoadModel.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="ctx">Model load context</param>
      internal TFStandardODScorerTransformer(IHostEnvironment env, ModelLoadContext ctx)
         : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TFStandardODScorerTransformer)))
      {
         Host.CheckValue(ctx, nameof(ctx));
         ctx.CheckAtModel(GetVersionInfo());
         outputClassesColumnName = ctx.LoadNonEmptyString();
         outputScoresColumnName = ctx.LoadNonEmptyString();
         outputBoxesColumnName = ctx.LoadNonEmptyString();
         outputLabelsColumnName = ctx.LoadNonEmptyString();
         inputClassesColumnName = ctx.LoadNonEmptyString();
         inputScoresColumnName = ctx.LoadNonEmptyString();
         inputBoxesColumnName = ctx.LoadNonEmptyString();
         minScore = float.Parse(ctx.LoadNonEmptyString(), CultureInfo.InvariantCulture);
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
      internal static IDataTransform Create(IHostEnvironment env, TFStandardODScorerEstimator.Options options, IDataView input)
      {
         Contracts.CheckValue(env, nameof(env));
         env.CheckValue(options, nameof(options));
         env.CheckValue(input, nameof(input));
         env.CheckValue(options.OutputClassesColumnName, nameof(options.OutputClassesColumnName));
         env.CheckValue(options.OutputScoresColumnName, nameof(options.OutputScoresColumnName));
         env.CheckValue(options.OutputBoxesColumnName, nameof(options.OutputBoxesColumnName));
         env.CheckValue(options.OutputLabelsColumnName, nameof(options.OutputLabelsColumnName));
         env.CheckValue(options.InputClassesColumnName, nameof(options.InputClassesColumnName));
         env.CheckValue(options.InputScoresColumnName, nameof(options.InputScoresColumnName));
         env.CheckValue(options.InputBoxesColumnName, nameof(options.InputBoxesColumnName));
         return new TFStandardODScorerEstimator(env, options).Fit(input).MakeDataTransform(input);
      }
      /// <summary>
      /// Factory method for SignatureLoadDataTransform.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="ctx">Load context</param>
      /// <param name="input">Input data</param>
      private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
          => new TFStandardODScorerTransformer(env, ctx).MakeDataTransform(input);
      /// <summary>
      /// Factory method for SignatureLoadRowMapper.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="ctx">Load context</param>
      /// <param name="inputSchema">Input schema</param>
      private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
          => new TFStandardODScorerTransformer(env, ctx).MakeRowMapper(inputSchema);
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
            modelSignature: "TFSTODSC", // TensorFlow object detection scorer transformer (max 8 chars)
            verWrittenCur: 0x00010001,  // Initial
            verReadableCur: 0x00010001,
            verWeCanReadBack: 0x00010001,
            loaderSignature: LoaderSignature,
            loaderAssemblyName: typeof(TFStandardODScorerTransformer).Assembly.FullName);
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
         Host.AssertNonWhiteSpace(outputClassesColumnName);
         Host.AssertNonWhiteSpace(outputScoresColumnName);
         Host.AssertNonWhiteSpace(outputBoxesColumnName);
         Host.AssertNonWhiteSpace(outputLabelsColumnName);
         ctx.SaveNonEmptyString(outputClassesColumnName);
         ctx.SaveNonEmptyString(outputScoresColumnName);
         ctx.SaveNonEmptyString(outputBoxesColumnName);
         ctx.SaveNonEmptyString(outputLabelsColumnName);
         ctx.SaveNonEmptyString(inputClassesColumnName);
         ctx.SaveNonEmptyString(inputScoresColumnName);
         ctx.SaveNonEmptyString(inputBoxesColumnName);
         ctx.SaveString(minScore.ToString(CultureInfo.InvariantCulture));
         ctx.Writer.Write(labels?.Length ?? 0);
         if (labels != null) {
            foreach (var label in labels)
               ctx.SaveNonEmptyString(label);
         }
      }
      #endregion
   }

   partial class TFStandardODScorerTransformer // Mapper
   {
      /// <summary>
      /// The mapper for <see cref="TFStandardODScorerTransformer"/>
      /// </summary>
      private sealed partial class Mapper : MapperBase
      {
         #region Fields
         /// <summary>
         /// Cache for outputs
         /// </summary>
         private readonly Cache cache;
         /// <summary>
         /// Index of the input column for the detection boxes
         /// </summary>
         private readonly int inputBoxesColumnIndex;
         /// <summary>
         /// Index of the input column for the detection classes
         /// </summary>
         private readonly int inputClassesColumnIndex;
         /// <summary>
         /// Index of the input column for the detection scores
         /// </summary>
         private readonly int inputScoresColumnIndex;
         /// <summary>
         /// The associated transformer
         /// </summary>
         private readonly TFStandardODScorerTransformer transformer;
         #endregion
         #region Methods
         /// <summary>
         /// Constructor
         /// </summary>
         /// <param name="transformer">The associated transformer</param>
         /// <param name="inputSchema"></param>
         public Mapper(TFStandardODScorerTransformer transformer, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(transformer, nameof(transformer)).Host.Register(nameof(Mapper)), inputSchema, transformer)
         {
            Host.CheckValue(transformer, nameof(transformer));
            this.transformer = transformer;
            // Check presence of input columns.
            if (!inputSchema.TryGetColumnIndex(this.transformer.inputBoxesColumnName, out inputBoxesColumnIndex))
               throw Host.ExceptSchemaMismatch(nameof(inputSchema), "source", this.transformer.outputBoxesColumnName);
            if (!inputSchema.TryGetColumnIndex(this.transformer.inputClassesColumnName, out inputClassesColumnIndex))
               throw Host.ExceptSchemaMismatch(nameof(inputSchema), "source", this.transformer.outputClassesColumnName);
            if (!inputSchema.TryGetColumnIndex(this.transformer.inputScoresColumnName, out inputScoresColumnIndex))
               throw Host.ExceptSchemaMismatch(nameof(inputSchema), "source", this.transformer.outputScoresColumnName);
            // Create the cache
            cache = new Cache(this);
         }
         /// <summary>
         /// Get the dependencies
         /// </summary>
         /// <param name="activeOutput">Active output indicator</param>
         /// <returns>The dependency functions</returns>
         private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput) =>
            col => Enumerable.Range(0, 4).Any(i => activeOutput(i)) && new[] { inputBoxesColumnIndex, inputClassesColumnIndex, inputScoresColumnIndex }.Any(ix => ix == col);
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

   partial class TFStandardODScorerTransformer // Cached values for the mapper
   {
      partial class Mapper
      {
         /// <summary>
         /// The cache for the mapper <see cref="Mapper"/>
         /// </summary>
         private class Cache
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
                  // Store the current position
                  Position = input.Position;
                  // Get the predictions from the model
                  var classesTensor = default(VBuffer<float>);
                  input.GetGetter<VBuffer<float>>(input.Schema[mapper.inputClassesColumnIndex]).Invoke(ref classesTensor);
                  var classesData = classesTensor.Items().Select(item => (int)item.Value).ToArray();
                  var scoresTensor = default(VBuffer<float>);
                  input.GetGetter<VBuffer<float>>(input.Schema[mapper.inputScoresColumnIndex]).Invoke(ref scoresTensor);
                  var boxesTensor = default(VBuffer<float>);
                  input.GetGetter<VBuffer<float>>(input.Schema[mapper.inputBoxesColumnIndex]).Invoke(ref boxesTensor);
                  // Filter or take the predictions as they are
                  if (mapper.transformer.minScore <= 0.0) {
                     // Store the results as they are
                     Boxes = boxesTensor;
                     Classes = new VBuffer<int>(classesData.Length, classesData);
                     Scores = scoresTensor;
                  }
                  else {
                     // Filter items lower that threshold
                     var filteredClasses = new List<int>();
                     var filteredBoxes = new List<float>();
                     var filteredScores = new List<float>();
                     var classesEnumerator = classesData.AsEnumerable().GetEnumerator();
                     var boxesEnumerator = boxesTensor.GetValues().GetEnumerator();
                     var scoresEnumerator = scoresTensor.GetValues().GetEnumerator();
                     while (scoresEnumerator.MoveNext()) {
                        classesEnumerator.MoveNext();
                        boxesEnumerator.MoveNext();
                        if (scoresEnumerator.Current < mapper.transformer.minScore) {
                           boxesEnumerator.MoveNext();
                           boxesEnumerator.MoveNext();
                           boxesEnumerator.MoveNext();
                        }
                        else {
                           filteredClasses.Add(classesEnumerator.Current);
                           filteredBoxes.Add(boxesEnumerator.Current);
                           boxesEnumerator.MoveNext();
                           filteredBoxes.Add(boxesEnumerator.Current);
                           boxesEnumerator.MoveNext();
                           filteredBoxes.Add(boxesEnumerator.Current);
                           boxesEnumerator.MoveNext();
                           filteredBoxes.Add(boxesEnumerator.Current);
                           filteredScores.Add(scoresEnumerator.Current);
                        }
                     }
                     // Store the filtered results
                     Boxes = new VBuffer<float>(filteredBoxes.Count, filteredBoxes.ToArray());
                     Classes = new VBuffer<int>(filteredClasses.Count, filteredClasses.ToArray());
                     Scores = new VBuffer<float>(filteredScores.Count, filteredScores.ToArray());
                  }
                  // Labels in text format
                  var labelsData = default(ReadOnlyMemory<char>[]);
                  if (mapper.transformer.labels != null) {
                     labelsData = Classes.Items().Select(item =>
                        (item.Value > -1 && item.Value < mapper.transformer.labels.Length ? mapper.transformer.labels[item.Value] : item.Value.ToString()).AsMemory())
                        .ToArray();
                  }
                  else
                     labelsData = Classes.Items().Select(item => item.Value.ToString().AsMemory()).ToArray();
                  Labels = new VBuffer<ReadOnlyMemory<char>>(labelsData.Length, labelsData);
               }
               return this;
            }
         }
      }
   }

   /// <summary>
   /// Estimator for the <see cref="TFStandardODScorerTransformer"/>.
   /// </summary>
   public sealed partial class TFStandardODScorerEstimator : TrivialEstimator<TFStandardODScorerTransformer>
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
      internal TFStandardODScorerEstimator(IHostEnvironment env, Options options)
         : base(
            Contracts.CheckRef(env, nameof(env)).Register(nameof(TFStandardODScorerEstimator)),
            new TFStandardODScorerTransformer(
               env,
               options.OutputClassesColumnName,
               options.OutputScoresColumnName,
               options.OutputBoxesColumnName,
               options.OutputLabelsColumnName,
               options.InputClassesColumnName,
               options.InputScoresColumnName,
               options.InputBoxesColumnName,
               options.MinScore,
               options.Labels))
      {
         Host.CheckNonEmpty(options.OutputClassesColumnName, nameof(options.OutputClassesColumnName));
         Host.CheckNonEmpty(options.OutputScoresColumnName, nameof(options.OutputScoresColumnName));
         Host.CheckNonEmpty(options.OutputBoxesColumnName, nameof(options.OutputBoxesColumnName));
         Host.CheckNonEmpty(options.OutputLabelsColumnName, nameof(options.OutputLabelsColumnName));
         this.options.InputClassesColumnName = !string.IsNullOrWhiteSpace(options.InputClassesColumnName) ? options.InputClassesColumnName : options.OutputClassesColumnName;
         this.options.InputScoresColumnName = !string.IsNullOrWhiteSpace(options.InputScoresColumnName) ? options.InputScoresColumnName : options.OutputScoresColumnName;
         this.options.InputBoxesColumnName = !string.IsNullOrWhiteSpace(options.InputBoxesColumnName) ? options.InputBoxesColumnName : options.OutputBoxesColumnName;
         this.options.Labels = options.Labels;
         this.options.MinScore = options.MinScore;
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
         if (!inputSchema.TryFindColumn(options.InputClassesColumnName, out var col))
            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", options.InputClassesColumnName);
         if ((col.Kind != SchemaShape.Column.VectorKind.VariableVector && col.Kind != SchemaShape.Column.VectorKind.Vector) || col.ItemType != NumberDataViewType.Single)
            throw Host.ExceptSchemaMismatch(nameof(options.InputClassesColumnName), "input", options.InputClassesColumnName, "Vector of Single", col.GetTypeString());
         if (!inputSchema.TryFindColumn(options.InputScoresColumnName, out col))
            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", options.InputScoresColumnName);
         if ((col.Kind != SchemaShape.Column.VectorKind.VariableVector && col.Kind != SchemaShape.Column.VectorKind.Vector) || col.ItemType != NumberDataViewType.Single)
            throw Host.ExceptSchemaMismatch(nameof(options.InputScoresColumnName), "input", options.InputScoresColumnName, "Vector of Single", col.GetTypeString());
         if (!inputSchema.TryFindColumn(options.InputBoxesColumnName, out col))
            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", options.InputBoxesColumnName);
         if ((col.Kind != SchemaShape.Column.VectorKind.VariableVector && col.Kind != SchemaShape.Column.VectorKind.Vector) || col.ItemType != NumberDataViewType.Single)
            throw Host.ExceptSchemaMismatch(nameof(options.InputBoxesColumnName), "input", options.InputBoxesColumnName, "Vector of Single", col.GetTypeString());
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
             SchemaShape.Column.VectorKind.VariableVector,
             NumberDataViewType.Single,
             false);
         result[options.OutputClassesColumnName] = new SchemaShape.Column(
             options.OutputClassesColumnName,
             SchemaShape.Column.VectorKind.VariableVector,
             NumberDataViewType.Int32,
             false,
             annotation);
         result[options.OutputLabelsColumnName] = new SchemaShape.Column(
             options.OutputLabelsColumnName,
             SchemaShape.Column.VectorKind.VariableVector,
             TextDataViewType.Instance,
             false);
         result[options.OutputScoresColumnName] = new SchemaShape.Column(
             options.OutputScoresColumnName,
             SchemaShape.Column.VectorKind.VariableVector,
             NumberDataViewType.Single,
             false);
         return new SchemaShape(result.Values);
      }
      #endregion
   }

   partial class TFStandardODScorerEstimator // Options
   {
      /// <summary>
      /// The options for the <see cref="TFStandardODScorerTransformer"/>.
      /// </summary>
      public sealed class Options : TransformInputBase
      {
         #region Fields
         /// <summary>
         /// Default name for the classes column
         /// </summary>
         public const string DefaultClassesColumnName = "detection_classes";
         /// <summary>
         /// Default name for the scores column
         /// </summary>
         public const string DefaultScoresColumnName = "detection_scores";
         /// <summary>
         /// Default name for the boxes column
         /// </summary>
         public const string DefaultBoxesColumnName = "detection_boxes";
         /// <summary>
         /// Default name for the output labels column
         /// </summary>
         public const string DefaultLabelsColumnName = "detection_labels";
         /// <summary>
         /// Default minimum score for detections
         /// </summary>
         public const float DefaultMinScore = 0.2f;
         /// <summary>
         /// The name of the input column containing the detection classes.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input column with detection classes", Name = "InputClasses", ShortName = "ic", SortOrder = 1)]
         public string InputClassesColumnName = DefaultClassesColumnName;
         /// <summary>
         /// The name of the input column containing the detection scores.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input column with detection scores", Name = "InputScores", ShortName = "is", SortOrder = 2)]
         public string InputScoresColumnName = DefaultScoresColumnName;
         /// <summary>
         /// The name of the input column containing the detection boxes.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input column with detection boxes", Name = "InputBoxes", ShortName = "ib", SortOrder = 3)]
         public string InputBoxesColumnName = DefaultBoxesColumnName;
         /// <summary>
         /// The name of the output column containing the detection classes.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the output column with detection classes", Name = "OutputClasses", ShortName = "oc", SortOrder = 4)]
         public string OutputClassesColumnName = DefaultClassesColumnName;
         /// <summary>
         /// The name of the output column containing the detection scores.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the output column with detection scores", Name = "OutputScores", ShortName = "os", SortOrder = 5)]
         public string OutputScoresColumnName = DefaultScoresColumnName;
         /// <summary>
         /// The name of the output column containing the detection boxes.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the output column with detection boxes", Name = "OutputBoxes", ShortName = "ob", SortOrder = 6)]
         public string OutputBoxesColumnName = DefaultBoxesColumnName;
         /// <summary>
         /// The name of the output column containing the detection labels.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the output column with detection labels", Name = "OutputLabels", ShortName = "ol", SortOrder = 7)]
         public string OutputLabelsColumnName = DefaultLabelsColumnName;
         /// <summary>
         /// Minimum score for detections
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Minimum score for detections", Name = "MinScore", ShortName = "ms", SortOrder = 8)]
         public float MinScore = DefaultMinScore;
         /// <summary>
         /// Set of labels
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "List of labels associated to the model", Name = "Labels", ShortName = "l", SortOrder = 10)]
         public IEnumerable<string> Labels = null;
         #endregion
      }
   }
}

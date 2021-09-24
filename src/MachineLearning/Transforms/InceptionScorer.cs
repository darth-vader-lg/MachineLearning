using MachineLearning.Transforms;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;

[assembly: LoadableClass(InceptionScorerTransformer.Summary, typeof(IDataTransform), typeof(InceptionScorerTransformer), typeof(InceptionScorerEstimator.Options), typeof(SignatureDataTransform),
    InceptionScorerTransformer.UserName)]

[assembly: LoadableClass(InceptionScorerTransformer.Summary, typeof(IDataTransform), typeof(InceptionScorerTransformer), null, typeof(SignatureLoadDataTransform),
    InceptionScorerTransformer.UserName, InceptionScorerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(InceptionScorerTransformer), null, typeof(SignatureLoadModel),
    InceptionScorerTransformer.UserName, InceptionScorerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(InceptionScorerTransformer), null, typeof(SignatureLoadRowMapper),
    InceptionScorerTransformer.UserName, InceptionScorerTransformer.LoaderSignature)]

namespace MachineLearning.Transforms
{
   /// <summary>
   /// Transformer for the output of the Inception V1 CNN
   /// </summary>
   public sealed partial class InceptionScorerTransformer : RowToRowTransformerBase
   {
      #region Fields
      /// <summary>
      /// Name of the input column (the output of the Inception model)
      /// </summary>
      private readonly string inputColumnName;
      /// <summary>
      /// Set of the labels associate with the model
      /// </summary>
      private readonly string[] labels;
      /// <summary>
      /// Signature for the loader
      /// </summary>
      internal const string LoaderSignature = "InceptionScorer";
      /// <summary>
      /// Name of the predicted scores column
      /// </summary>
      private readonly string scoreColumnName;
      /// <summary>
      /// Name of the predicted label column
      /// </summary>
      private readonly string predictedLabelColumnName;
      /// <summary>
      /// Summary for the loader
      /// </summary>
      internal const string Summary = "Transform the data of an Inception CNN";
      /// <summary>
      /// User name for the loader
      /// </summary>
      internal const string UserName = "InceptionScorer";
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="inputColumnName">Name of the input column (the output of the Inception model)</param>
      /// <param name="predictedLabelColumnName">Name of the predicted label column</param>
      /// <param name="scoreColumnName">Name of the predicted scores column</param>
      /// <param name="labels">Set of the labels associate with the model</param>
      internal InceptionScorerTransformer(
         IHostEnvironment env,
         string inputColumnName = InceptionScorerEstimator.Options.DefaultInputColumnName,
         string predictedLabelColumnName = DefaultColumnNames.PredictedLabel,
         string scoreColumnName = DefaultColumnNames.Score,
         IEnumerable<string> labels = null)
         : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(InceptionScorerTransformer)))
      {
         Host.CheckValue(predictedLabelColumnName, nameof(predictedLabelColumnName));
         Host.CheckValue(inputColumnName, nameof(inputColumnName));
         Host.CheckValue(scoreColumnName, nameof(scoreColumnName));
         this.predictedLabelColumnName = predictedLabelColumnName;
         this.inputColumnName = inputColumnName;
         this.scoreColumnName = scoreColumnName;
         this.labels = labels?.ToArray();
      }
      /// <summary>
      /// Factory method for SignatureLoadModel.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="ctx">Model load context</param>
      internal InceptionScorerTransformer(IHostEnvironment env, ModelLoadContext ctx)
         : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(InceptionScorerTransformer)))
      {
         Host.CheckValue(ctx, nameof(ctx));
         ctx.CheckAtModel(GetVersionInfo());
         predictedLabelColumnName = ctx.LoadNonEmptyString();
         inputColumnName = ctx.LoadNonEmptyString();
         scoreColumnName = ctx.LoadNonEmptyString();
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
      internal static IDataTransform Create(IHostEnvironment env, InceptionScorerEstimator.Options options, IDataView input)
      {
         Contracts.CheckValue(env, nameof(env));
         env.CheckValue(options, nameof(options));
         env.CheckValue(input, nameof(input));
         env.CheckValue(options.InputColumnName, nameof(options.InputColumnName));
         env.CheckValue(options.PredictedLabelColumnName, nameof(options.PredictedLabelColumnName));
         env.CheckValue(options.ScoreColumnName, nameof(options.ScoreColumnName));
         return new InceptionScorerEstimator(env, options).Fit(input).MakeDataTransform(input);
      }
      /// <summary>
      /// Factory method for SignatureLoadDataTransform.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="ctx">Load context</param>
      /// <param name="input">Input data</param>
      private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
          => new InceptionScorerTransformer(env, ctx).MakeDataTransform(input);
      /// <summary>
      /// Factory method for SignatureLoadRowMapper.
      /// </summary>
      /// <param name="env">The host environment</param>
      /// <param name="ctx">Load context</param>
      /// <param name="inputSchema">Input schema</param>
      private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
          => new InceptionScorerTransformer(env, ctx).MakeRowMapper(inputSchema);
      /// <summary>
      /// Get the info about the version
      /// </summary>
      /// <returns></returns>
      private static VersionInfo GetVersionInfo()
      {
         return new VersionInfo(
            modelSignature: "INCV1SCR", // Inception V1 scorer transformer
            verWrittenCur: 0x00010001, // Initial
            verReadableCur: 0x00010001,
            verWeCanReadBack: 0x00010001,
            loaderSignature: LoaderSignature,
            loaderAssemblyName: typeof(InceptionScorerTransformer).Assembly.FullName);
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
         Host.AssertNonWhiteSpace(predictedLabelColumnName);
         Host.AssertNonWhiteSpace(inputColumnName);
         Host.AssertNonWhiteSpace(scoreColumnName);
         ctx.SaveNonEmptyString(predictedLabelColumnName);
         ctx.SaveNonEmptyString(inputColumnName);
         ctx.SaveNonEmptyString(scoreColumnName);
         ctx.Writer.Write(labels?.Length ?? 0);
         if (labels != null) {
            foreach (var label in labels)
               ctx.SaveNonEmptyString(label);
         }
      }
      #endregion
   }

   partial class InceptionScorerTransformer // Mapper
   {
      /// <summary>
      /// The mapper for <see cref="InceptionScorerTransformer"/>
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
         private readonly InceptionScorerTransformer transformer;
         #endregion
         /// <summary>
         /// Constructor
         /// </summary>
         /// <param name="transformer">The associated transformer</param>
         /// <param name="inputSchema"></param>
         public Mapper(InceptionScorerTransformer transformer, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(transformer, nameof(transformer)).Host.Register(nameof(Mapper)), inputSchema, transformer)
         {
            Host.CheckValue(transformer, nameof(transformer));
            this.transformer = transformer;
            // Check presence of input columns.
            if (!inputSchema.TryGetColumnIndex(this.transformer.inputColumnName, out inputColumnIndex))
               throw Host.ExceptSchemaMismatch(nameof(inputSchema), "source", this.transformer.inputColumnName);
            // Create the cache
            cache = new Cache(this);
         }
         /// <summary>
         /// Get the dependencies
         /// </summary>
         /// <param name="activeOutput">Active output indicator</param>
         /// <returns>The dependency functions</returns>
         private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
         {
            // Range goes from 0 to 1 because there is only one output column. @@@ Check if it's true from 0 to 1
            return col => Enumerable.Range(0, 1).Any(i => activeOutput(i)) && inputColumnIndex == col;
         }
         /// <summary>
         /// Get the output columns
         /// </summary>
         /// <returns>The array of output columns</returns>
         protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
         {
            // Build the annotations containing the slots (label names)
            var annotations = default(DataViewSchema.Annotations);
            if (transformer.labels != null) {
               var builder = new DataViewSchema.Annotations.Builder();
               builder.AddSlotNames(transformer.labels.Length, (ref VBuffer<ReadOnlyMemory<char>> value) =>
               {
                  value = new VBuffer<ReadOnlyMemory<char>>(transformer.labels.Length, (from label in transformer.labels
                                                                                        select label.AsMemory()).ToArray());
               });
               annotations = builder.ToAnnotations();
            }
            return new DataViewSchema.DetachedColumn[]
            {
               new DataViewSchema.DetachedColumn(transformer.predictedLabelColumnName, TextDataViewType.Instance),
               new DataViewSchema.DetachedColumn(transformer.scoreColumnName, new VectorDataViewType(NumberDataViewType.Single, transformer.labels?.Length ?? 0), annotations)
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
               0 => (ValueGetter<ReadOnlyMemory<char>>)((ref ReadOnlyMemory<char> value) => value = cache.Get(input).PredictedLabel),
               1 => (ValueGetter<VBuffer<float>>)((ref VBuffer<float> value) => value = cache.Get(input).Scores),
               _ => null,
            };
         }
         /// <summary>
         /// Save the model
         /// </summary>
         /// <param name="ctx">Save context</param>
         private protected override void SaveModel(ModelSaveContext ctx)
             => transformer.SaveModel(ctx);
      }
   }

   partial class InceptionScorerTransformer // Cached values for the mapper
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
            /// Row position of the cache
            /// </summary>
            internal long? Position { get; private set; }
            /// <summary>
            /// Cached label
            /// </summary>
            internal ReadOnlyMemory<char> PredictedLabel { get; private set; }
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
                  // Values
                  var scores = default(VBuffer<float>);
                  input.GetGetter<VBuffer<float>>(input.Schema[mapper.inputColumnIndex]).Invoke(ref scores);
                  // Create array of values from NN output
                  var values = new float[Math.Min(scores.Length, mapper.transformer.labels?.Length ?? scores.Length)];
                  var inputValues = scores.GetValues();
                  var inputEnumerator = inputValues.GetEnumerator();
                  // Fill the array and compute min, max and max score ix
                  var max = -float.MaxValue;
                  var maxScoreIx = -1;
                  for (var i = 0; i < values.Length && inputEnumerator.MoveNext(); i++) {
                     if (inputEnumerator.Current > max) {
                        max = inputEnumerator.Current;
                        maxScoreIx = i;
                     }
                     values[i] = Math.Max(0, inputEnumerator.Current);
                  }
                  // Normalize scores
                  if (max > 0 && Math.Abs(max - 1f) > 1E-3F) {
                     for (var i = 0; i < values.Length; i++)
                        values[i] /= max;
                  }
                  // Predicted label
                  if (maxScoreIx < mapper.transformer.labels?.Length)
                     PredictedLabel = mapper.transformer.labels[maxScoreIx].AsMemory();
                  else
                     PredictedLabel = maxScoreIx.ToString().AsMemory();
                  // Predicted scores
                  Scores = new VBuffer<float>(values.Length, values);
               }
               return this;
            }
         }
      }
   }

   /// <summary>
   /// Estimator for the <see cref="InceptionScorerTransformer"/>.
   /// </summary>
   public sealed partial class InceptionScorerEstimator : TrivialEstimator<InceptionScorerTransformer>
   {
      /// <summary>
      /// The input column name (the output of the scorer of the Inception V1 CNN)
      /// </summary>
      private readonly string inputColumnName;
      /// <summary>
      /// The name of the predicted label
      /// </summary>
      private readonly string predictedLabelColumnName;
      /// <summary>
      /// The name of the scores vector containing also the slot names
      /// </summary>
      private readonly string scoreColumnName;
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="env">Host environment</param>
      /// <param name="options">Scorer options</param>
      internal InceptionScorerEstimator(IHostEnvironment env, Options options)
         : base(
            Contracts.CheckRef(env, nameof(env)).Register(nameof(InceptionScorerEstimator)),
            new InceptionScorerTransformer(env, options.InputColumnName, options.PredictedLabelColumnName, options.ScoreColumnName, options.Labels))
      {
         Host.CheckNonEmpty(options.InputColumnName, nameof(options.InputColumnName));
         Host.CheckNonEmpty(options.PredictedLabelColumnName, nameof(options.PredictedLabelColumnName));
         Host.CheckNonEmpty(options.ScoreColumnName, nameof(options.ScoreColumnName));
         inputColumnName = options.InputColumnName;
         predictedLabelColumnName = options.PredictedLabelColumnName;
         scoreColumnName = options.ScoreColumnName;
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
         if (!inputSchema.TryFindColumn(inputColumnName, out var col))
            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName);
         if (col.Kind != SchemaShape.Column.VectorKind.Vector || col.ItemType != NumberDataViewType.Single)
            throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName, "Vector of Single", col.GetTypeString());
         result[predictedLabelColumnName] = new SchemaShape.Column(
             predictedLabelColumnName,
             SchemaShape.Column.VectorKind.Scalar,
             TextDataViewType.Instance,
             false);
         result[scoreColumnName] = new SchemaShape.Column(
             scoreColumnName,
             SchemaShape.Column.VectorKind.Vector,
             NumberDataViewType.Single,
             false);
         return new SchemaShape(result.Values);
      }
   }

   partial class InceptionScorerEstimator // Options
   {
      /// <summary>
      /// The options for the <see cref="InceptionScorerTransformer"/>.
      /// </summary>
      public sealed class Options : TransformInputBase
      {
         #region Fields
         /// <summary>
         /// Default name for the input column
         /// </summary>
         public const string DefaultInputColumnName = "logits";
         /// <summary>
         /// The name of the column containing the inputs for the model.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the input column", Name = "Source", ShortName = "s", SortOrder = 1)]
         public string InputColumnName = DefaultInputColumnName;
         /// <summary>
         /// Name of the column that will contain the predicted label from output scores.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the column that will contain the predicted label", Name = "PredictedLabel", ShortName = "l", SortOrder = 2)]
         public string PredictedLabelColumnName = DefaultColumnNames.PredictedLabel;
         /// <summary>
         /// Name of the column that will contain the vector of scores and slot names in the annotation.
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the column that will contain the scores for each category", Name = "Scores", ShortName = "ss", SortOrder = 3)]
         public string ScoreColumnName = DefaultColumnNames.Score;
         /// <summary>
         /// Set of labels
         /// </summary>
         [Argument(ArgumentType.AtMostOnce, HelpText = "List of labels associated to the model", Name = "Labels", ShortName = "l", SortOrder = 4)]
         public IEnumerable<string> Labels = null;
         #endregion
      }
   }
}

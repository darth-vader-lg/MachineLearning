using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using System;
using System.Linq;

namespace MachineLearning.Transforms
{
   /// <summary>
   /// Estimator di aggiunta costanti
   /// </summary>
   public sealed class ConstantValueEstimator : IEstimator<ConstantValueTransformer>
   {
      #region Properties
      /// <summary>
      /// Nome colonna di output
      /// </summary>
      public string Value { get; }
      /// <summary>
      /// Nome colonna di output
      /// </summary>
      public string OutputColumnName { get; }
      /// <summary>
      /// Catalogo trasformazioni
      /// </summary>
      private TransformsCatalog Transforms { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="transformCatalog">Catalogo di trasformazioni</param>
      /// <param name="outputColumnName">Nome colonna di output</param>
      /// <param name="value">Valore costante</param>
      internal ConstantValueEstimator(TransformsCatalog transformCatalog, string outputColumnName, string value)
      {
         Transforms = transformCatalog;
         Value = value;
         OutputColumnName = outputColumnName;
      }
      /// <summary>
      /// Ottiene il transformer
      /// </summary>
      /// <param name="input">Dati di input</param>
      /// <returns>Il transformer</returns>
      public ConstantValueTransformer Fit(IDataView input) => new(Transforms, OutputColumnName, Value, input.Schema);
      /// <summary>
      /// Restituisce lo schema di output dell'estimatore
      /// </summary>
      /// <param name="inputSchema">Lo schema di input</param>
      /// <returns>Lo schema di output</returns>
      public SchemaShape GetOutputSchema(SchemaShape inputSchema) => new ConstantValueTransformer(Transforms, OutputColumnName, Value).Pipe.GetOutputSchema(inputSchema);
      #endregion
   }

   /// <summary>
   /// Transformer per l'aggiunta di colonne con valori costanti
   /// </summary>
   public sealed partial class ConstantValueTransformer : TransformerBase
   {
      #region Properties
      /// <summary>
      /// Schema di input
      /// </summary>
      public sealed override DataViewSchema InputSchema { get; }
      /// <summary>
      /// Pipe di trasformazione
      /// </summary>
      internal sealed override IEstimator<ITransformer> Pipe { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="transformsCatalog">Catalogo di trasformazioni</param>
      /// <param name="outputColumnName">Nome colonna di output</param>
      /// <param name="value">Valore costante</param>
      /// <param name="inputSchema">Schema di input</param>
      internal ConstantValueTransformer(TransformsCatalog transformsCatalog, string outputColumnName, string value, DataViewSchema inputSchema = null) : base(transformsCatalog)
      {
         Pipe =
            transformsCatalog.CustomMapping(new Action<Mapper.Input, Mapper.Output>((In, Out) => { }), nameof(Mapper.Output.CE6C5270_FCDA_44F1_8680_7C5BF491B2B1))
            .Append(transformsCatalog.Expression(outputColumnName, $"i:{value}", nameof(Mapper.Output.CE6C5270_FCDA_44F1_8680_7C5BF491B2B1)))
            .Append(transformsCatalog.DropColumns(nameof(Mapper.Output.CE6C5270_FCDA_44F1_8680_7C5BF491B2B1)));
         InputSchema = inputSchema ?? DataViewSchemaBuilder.Build(Array.Empty<(string Name, Type Type)>());
      }
      #endregion
   }

   public sealed partial class ConstantValueTransformer
   {
      /// <summary>
      /// Mapper per l'assegnazione delle costanti
      /// </summary>
      [CustomMappingFactoryAttribute(nameof(Output.CE6C5270_FCDA_44F1_8680_7C5BF491B2B1))]
      internal class Mapper : CustomMappingFactory<Mapper.Input, Mapper.Output>
      {
         #region Properties
         /// <summary>
         /// Dati di input (vuoti)
         /// </summary>
         internal class Input { }
         /// <summary>
         /// Dati di output (solo come placeholder)
         /// </summary>
         internal class Output { public int CE6C5270_FCDA_44F1_8680_7C5BF491B2B1 { get; set; } }
         #endregion
         #region Methods
         /// <summary>
         /// Azione di mappatura
         /// </summary>
         /// <returns></returns>
         public override Action<Input, Output> GetMapping() => new((input, output) => { });
         #endregion
      }
   }
}

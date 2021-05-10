using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;

namespace MachineLearning.Transforms
{
   /// <summary>
   /// Classe base per i transformers
   /// </summary>
   public abstract class TransformerBase : ITransformer
   {
      #region Fields
      /// <summary>
      /// Transformer
      /// </summary>
      private ITransformer transformer;
      #endregion
      #region Properties
      /// <summary>
      /// Lo schema di input
      /// </summary>
      public abstract DataViewSchema InputSchema { get; }
      /// <summary>
      /// Indicatore di trasformatore riga a riga
      /// </summary>
      public bool IsRowToRowMapper => Transformer.IsRowToRowMapper;
      /// <summary>
      /// La pipe di trasformazioni
      /// </summary>
      internal abstract IEstimator<ITransformer> Pipe { get; }
      /// <summary>
      /// Il transformer
      /// </summary>
      private ITransformer Transformer
      {
         get
         {
            if (transformer != null)
               return transformer;
            var dataView = DataViewGrid.Create(Transforms.GetChannelProvider(), InputSchema);
            return transformer = Pipe.Fit(dataView);
         }
      }
      /// <summary>
      /// Catalogo trasformazioni
      /// </summary>
      protected TransformsCatalog Transforms { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="transformsCatalog">Catalogo di trasformazini</param>
      internal TransformerBase(TransformsCatalog transformsCatalog) => Transforms = transformsCatalog;
      /// <summary>
      /// Restituisce lo schema di output
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Lo schema di output</returns>
      public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => Transformer.GetOutputSchema(inputSchema);
      /// <summary>
      /// Restituisce il mappatore riga a riga
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Il mappatore</returns>
      public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => Transformer.GetRowToRowMapper(inputSchema);
      /// <summary>
      /// Effettua il salvataggio
      /// </summary>
      /// <param name="ctx">Contesto</param>
      public void Save(ModelSaveContext ctx) => Transformer.Save(ctx);
      /// <summary>
      /// Effettua la trasformazione dei dati
      /// </summary>
      /// <param name="input">Dati di input</param>
      /// <returns>I dati trasformati</returns>
      public IDataView Transform(IDataView input) => Transformer.Transform(input);
      #endregion
   }
}

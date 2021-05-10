﻿using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System.Threading;

namespace MachineLearning.Data
{
   /// <summary>
   /// Transformer di dati ML.NET
   /// </summary>
   public class DataTransformerMLNet : IDataTransformer, ITransformer
   {
      #region Properties
      /// <summary>
      /// Il transformer
      /// </summary>
      public ITransformer Transformer { get; }
      /// <summary>
      /// Definisce se e' un mapper riga a riga
      /// </summary>
      public bool IsRowToRowMapper => Transformer.IsRowToRowMapper;
      #endregion
      #region Methods
      public DataTransformerMLNet(IContextProvider<MLContext> context, ITransformer transformer)
      {
         MachineLearningContext.CheckContext(context, nameof(context));
         context.CheckValue(transformer, nameof(transformer));
         Transformer = transformer;
      }
      /// <summary>
      /// Restituisce lo schema di output
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Lo schema di output</returns>
      public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => Transformer.GetOutputSchema(inputSchema);
      /// <summary>
      /// Restituisce il mapper riga a riga
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Il mapper</returns>
      public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => Transformer.GetRowToRowMapper(inputSchema);
      /// <summary>
      /// Salva il transformer
      /// </summary>
      /// <param name="ctx">Contesto</param>
      public void Save(ModelSaveContext ctx) => Transformer.Save(ctx);
      /// <summary>
      /// Trasforma i dati
      /// </summary>
      /// <param name="data">Dati in ingresso</param>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      /// <returns>I dati trasformati</returns>
      public IDataAccess Transform(IDataAccess data, CancellationToken cancellation = default) => new DataAccess(data, Transformer.Transform(data));
      /// <summary>
      /// Esegue la trasformazione di dati
      /// </summary>
      /// <param name="input">Dati di input</param>
      /// <returns>I dati trasformati</returns>
      public IDataView Transform(IDataView input) => Transformer.Transform(input);
      #endregion
   }
}
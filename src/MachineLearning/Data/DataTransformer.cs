using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Diagnostics;
using System.Threading;

namespace MachineLearning.Data
{
   /// <summary>
   /// Transformer di dati
   /// </summary>
   public class DataTransformer<T> : IDataTransformer, IDisposable, ITransformer where T : class
   {
      #region Fields
      /// <summary>
      /// Indicatore di oggetto disposto
      /// </summary>
      private bool disposedValue;
      #endregion
      #region Properties
      /// <summary>
      /// Il transformer
      /// </summary>
      public virtual ITransformer Transformer { get; protected set; }
      /// <summary>
      /// Definisce se e' un mapper riga a riga
      /// </summary>
      public bool IsRowToRowMapper => Transformer.IsRowToRowMapper;
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="context">Context</param>
      protected DataTransformer(IExceptionContext context) => Contracts.CheckValue(context, nameof(context));
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="context">Context</param>
      /// <param name="transformer">Associated transformer</param>
      public DataTransformer(IExceptionContext context, ITransformer transformer) : this(context)
      {
         context.CheckValue(transformer, nameof(transformer));
         Transformer = transformer;
      }
      /// <summary>
      /// Implementazione della IDisposable
      /// </summary>
      public void Dispose()
      {
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Indicatore di dispose da codice</param>
      protected virtual void Dispose(bool disposing)
      {
         if (!disposedValue) {
            try {
               (Transformer as IDisposable)?.Dispose();
            }
            catch (Exception exc) {
               Trace.WriteLine(exc);
            }
            Transformer = null;
            disposedValue = true;
         }
      }
      /// <summary>
      /// Restituisce lo schema di output
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Lo schema di output</returns>
      public DataSchema GetOutputSchema(DataSchema inputSchema) => Transformer.GetOutputSchema(inputSchema);
      /// <summary>
      /// Restituisce il mapper riga a riga
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Il mapper</returns>
      public IRowToRowMapper GetRowToRowMapper(DataSchema inputSchema) => Transformer.GetRowToRowMapper(inputSchema);
      /// <summary>
      /// Get the output schema given the input schema
      /// </summary>
      /// <param name="inputSchema">The input schema</param>
      /// <returns>The output schema</returns>
      DataViewSchema ITransformer.GetOutputSchema(DataViewSchema inputSchema) => Transformer.GetOutputSchema(inputSchema);
      /// <summary>
      /// Get the row to row mapper fo the transofrmer
      /// </summary>
      /// <param name="inputSchema">Input schema</param>
      /// <returns>The row to row mapper</returns>
      IRowToRowMapper ITransformer.GetRowToRowMapper(DataViewSchema inputSchema) => Transformer.GetRowToRowMapper(inputSchema);
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

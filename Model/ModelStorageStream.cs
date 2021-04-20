using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;
using System.IO;

namespace MachineLearning.Model
{
   /// <summary>
   /// Gestore dello storage su stream dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageStream : IModelStorage, IDataTimestamp
   {
      #region Properties
      /// <summary>
      /// Funzione di ottenimento dello stream di lettura
      /// </summary>
      [field: NonSerialized]
      public Func<Stream> ReadStream { get; set; }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp { get; private set; } = DateTime.UtcNow;
      /// <summary>
      /// Funzione di ottenimento dello stream di scrittura
      /// </summary>
      [field: NonSerialized]
      public Func<Stream> WriteStream { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MLContext context, out DataViewSchema inputSchema)
      {
         Contracts.CheckValue(context, nameof(context));
         var stream = ReadStream?.Invoke();
         if (stream == null)
            throw new InvalidOperationException("Cannot read from the stream");
         return context.Model.Load(stream, out inputSchema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(MLContext context, ITransformer model, DataViewSchema inputSchema)
      {
         Contracts.CheckValue(context, nameof(context));
         var stream = WriteStream?.Invoke();
         if (stream == null)
            throw new InvalidOperationException("Cannot write to the stream");
         context.Model.Save(model, inputSchema, stream);
      }
      #endregion
   }
}

using MachineLearning.Data;
using Microsoft.ML;
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
      /// <typeparam name="T">Il tipo di contesto</typeparam>
      /// <param name="context">Contesto</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(IMachineLearningContext context, out DataViewSchema inputSchema)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         var stream = ReadStream?.Invoke();
         if (stream == null)
            throw new InvalidOperationException("Cannot read from the stream");
         return context.ML.NET.Model.Load(stream, out inputSchema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <typeparam name="T">Il tipo di contesto</typeparam>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(IMachineLearningContext context, ITransformer model, DataViewSchema inputSchema)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         var stream = WriteStream?.Invoke();
         if (stream == null)
            throw new InvalidOperationException("Cannot write to the stream");
         context.ML.NET.Model.Save(model, inputSchema, stream);
      }
      #endregion
   }
}

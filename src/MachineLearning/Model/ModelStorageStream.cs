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
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp { get; private set; }
      /// <summary>
      /// Eventuale path di importazione di un modello esterno (ONNX / TensorFlow, ecc...)
      /// </summary>
      public string ImportPath { get; set; }
      /// <summary>
      /// Funzione di ottenimento dello stream di lettura
      /// </summary>
      [field: NonSerialized]
      public Func<Stream> ReadStream { get; set; }
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
      public ITransformer LoadModel(MLContext context, out DataSchema inputSchema)
      {
         Contracts.CheckValue(context, nameof(context));
         var stream = ReadStream?.Invoke();
         if (stream == null)
            throw new InvalidOperationException("Cannot read from the stream");
         DataTimestamp = DateTime.UtcNow;
         var result = context.Model.Load(stream, out var mlSchema);
         inputSchema = mlSchema;
         return result;
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(MLContext context, ITransformer model, DataSchema inputSchema)
      {
         Contracts.CheckValue(context, nameof(context));
         var timestamp = DateTime.UtcNow;
         var stream = WriteStream?.Invoke();
         if (stream == null)
            throw new InvalidOperationException("Cannot write to the stream");
         context.Model.Save(model, inputSchema, stream);
         DataTimestamp = timestamp;
      }
      #endregion
   }
}

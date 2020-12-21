using Microsoft.ML;
using System;
using System.IO;

namespace MachineLearning
{
   /// <summary>
   /// Gestore dello storage su stream dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageStream : IModelStorage, ITimestamp
   {
      #region Properties
      [NonSerialized]
      /// <summary>
      /// Funzione di restituzione della stream in lettura
      /// </summary>
      private readonly Func<Stream> _readStreamGetter;
      /// <summary>
      /// Funzione di restituzione della stream in lettura
      /// </summary>
      [NonSerialized]
      private readonly Func<Stream> _writeStreamGetter;
      #endregion
      #region Properties
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime Timestamp { get; private set; } = DateTime.UtcNow;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ReadStreamGetter">Funzione di restituzione della stream in lettura</param>
      /// <param name="WriteStreamGetter">Funzione di restituzione della stream in scrittura</param>
      public ModelStorageStream(Func<Stream> ReadStreamGetter = null, Func<Stream> WriteStreamGetter = null)
      {
         this._readStreamGetter = ReadStreamGetter;
         this._writeStreamGetter = WriteStreamGetter;
      }
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MachineLearningContext ml, out DataViewSchema inputSchema)
      {
         if (_readStreamGetter == default) {
            inputSchema = default;
            return default;
         }
         using var stream = _readStreamGetter();
         return ml.NET.Model.Load(stream, out inputSchema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(MachineLearningContext ml, ITransformer model, DataViewSchema inputSchema)
      {
         lock (this) {
            var timestamp = DateTime.UtcNow;
            if (_writeStreamGetter == default)
               return;
            using var stream = _writeStreamGetter();
            ml.NET.Model.Save(model, inputSchema, stream);
            Timestamp = timestamp;
         }
      }
      #endregion
   }
}

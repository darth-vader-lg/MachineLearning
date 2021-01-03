using Microsoft.ML;
using System;
using System.IO;

namespace MachineLearning
{
   /// <summary>
   /// Classe per lo storage di dati di tipo testo su stream
   /// </summary>
   [Serializable]
   public sealed class DataStorageTextStream : DataStorageText, ITimestamp
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
      public DateTime Timestamp { get; private set; }
      /// <summary>
      /// Funzione di ottenimento dello stream di scrittura
      /// </summary>
      [field: NonSerialized]
      public Func<Stream> WriteStream { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Restituisce uno stream leggibile.
      /// </summary>
      /// <returns>Lo stream di lettura</returns>
      protected override Stream GetReadStream()
      {
         var stream = ReadStream?.Invoke();
         if (stream == null)
            throw new InvalidOperationException("Cannot read from the stream");
         return stream;
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      public override void SaveData(IMachineLearningContextProvider context, IDataView data)
      {
         var timestamp = DateTime.UtcNow;
         var stream = WriteStream?.Invoke();
         if (stream == null)
            throw new InvalidOperationException("Cannot write to the stream");
         lock (this) {
            SaveTextData(context, data, stream, SaveSchema, KeepHidden, ForceDense);
            Timestamp = timestamp;
         }
      }
      #endregion
   }
}

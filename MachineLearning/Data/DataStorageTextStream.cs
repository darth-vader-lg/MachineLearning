using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.IO;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo testo su stream
   /// </summary>
   [Serializable]
   public sealed class DataStorageTextStream : DataStorageText, IDataTimestamp
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
      public DateTime DataTimestamp { get; private set; }
      /// <summary>
      /// Funzione di ottenimento dello stream di scrittura
      /// </summary>
      [field: NonSerialized]
      public Func<Stream> WriteStream { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="options">Opzioni di caricamento testi</param>
      public DataStorageTextStream(TextLoader.Options options) : base(options) { }
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
         MachineLearningContext.CheckMLNET(context, nameof(context));
         var timestamp = DateTime.UtcNow;
         var stream = WriteStream?.Invoke();
         context.ML.NET.CheckIO(stream != null, "Cannot write to the stream");
         lock (this) {
            SaveTextData(context, data, stream, true);
            DataTimestamp = timestamp;
         }
      }
      #endregion
   }
}

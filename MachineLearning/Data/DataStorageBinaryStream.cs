using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.IO;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo binario su stream
   /// </summary>
   [Serializable]
   public sealed class DataStorageBinaryStream : DataStorageBinary, IDataTimestamp
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
      /// Restituisce uno stream leggibile.
      /// </summary>
      /// <returns>Lo stream di lettura</returns>
      protected override Stream GetReadStream() => ReadStream?.Invoke();
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="textLoaderOptions">Eventuali opzioni di caricamento testuale (non utilizzate per il salvataggio binario)</param>
      public override void SaveData(IMachineLearningContext context, IDataView data, TextLoader.Options textLoaderOptions = default)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         var timestamp = DateTime.UtcNow;
         var stream = WriteStream?.Invoke();
         context.ML.NET.CheckIO(stream != null, "Cannot write to the stream");
         lock (this) {
            SaveBinaryData(context, data, stream);
            DataTimestamp = timestamp;
         }
      }
      #endregion
   }
}

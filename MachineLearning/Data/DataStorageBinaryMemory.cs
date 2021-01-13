using Microsoft.ML;
using System;
using System.IO;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo binario in memoria
   /// </summary>
   [Serializable]
   public sealed class DataStorageBinaryMemory : DataStorageBinary, ITimestamp
   {
      #region Fields
      /// <summary>
      /// Dati testuali
      /// </summary>
      private byte[] _binaryData;
      #endregion
      #region Properties
      /// <summary>
      /// Dati testuali
      /// </summary>
      public byte[] BinaryData { get => _binaryData; set { _binaryData = value; Timestamp = DateTime.UtcNow; } }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime Timestamp { get; private set; } = DateTime.UtcNow;
      #endregion
      #region Methods
      /// <summary>
      /// Restituisce uno stream leggibile.
      /// </summary>
      /// <returns>Lo stream di lettura</returns>
      protected override Stream GetReadStream() => new MemoryStream(BinaryData);
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      public override void SaveData(IMachineLearningContextProvider context, IDataView data)
      {
         var timestamp = DateTime.UtcNow;
         lock (this) {
            var stream = new MemoryStream();
            SaveBinaryData(context, data, stream, KeepHidden);
            BinaryData = stream.ToArray();
            Timestamp = timestamp;
         }
      }
      #endregion
   }
}

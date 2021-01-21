using Microsoft.ML;
using System;
using System.IO;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo binario in memoria
   /// </summary>
   [Serializable]
   public sealed class DataStorageBinaryMemory : DataStorageBinary, IDataTimestamp
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
      public byte[] BinaryData { get => _binaryData; set { _binaryData = value; DataTimestamp = DateTime.UtcNow; } }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Restituisce uno stream leggibile.
      /// </summary>
      /// <returns>Lo stream di lettura</returns>
      protected override Stream GetReadStream() => BinaryData == null ? null : new MemoryStream(BinaryData);
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      public override void SaveData(IMachineLearningContextProvider context, IDataView data)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         var timestamp = DateTime.UtcNow;
         lock (this) {
            using var stream = new MemoryStream();
            SaveBinaryData(context, data, stream);
            BinaryData = stream.ToArray();
            stream.Close();
            DataTimestamp = timestamp;
         }
      }
      #endregion
   }
}

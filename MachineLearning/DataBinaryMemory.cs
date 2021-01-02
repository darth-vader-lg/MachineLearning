using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace MachineLearning
{
   /// <summary>
   /// Classe per lo storage di dati di tipo binario in memoria
   /// </summary>
   [Serializable]
   public sealed class DataBinaryMemory : IDataStorage, IMultiStreamSource, ITimestamp
   {
      #region Fields
      /// <summary>
      /// Dati testuali
      /// </summary>
      private byte[] _binaryData;
      #endregion
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => 1;
      /// <summary>
      /// Dati testuali
      /// </summary>
      public byte[] BinaryData { get => _binaryData; set { _binaryData = value; Timestamp = DateTime.UtcNow; } }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime Timestamp { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public DataBinaryMemory() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">Dati</param>
      public DataBinaryMemory(object context, IDataView data) => SaveData(context, data);
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">Dati</param>
      public DataBinaryMemory(object context, byte[] data)
      {
         BinaryData = data;
         SaveData(context, LoadData(context));
      }
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Il path</returns>
      string IMultiStreamSource.GetPathOrNull(int index) => null;
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      Stream IMultiStreamSource.Open(int index) => new MemoryStream(BinaryData ?? Array.Empty<byte>());
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      TextReader IMultiStreamSource.OpenTextReader(int index) => new StreamReader(((IMultiStreamSource)this).Open(index));
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(object context)
      {
         var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
         return ml.Data.LoadFromBinary(this);
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="schema">Commento contenente lo schema nei dati di tipo file testuali (ignorato negli altri)</param>
      public void SaveData(object context, IDataView data, bool schema = false)
      {
         lock (this) {
            // Contesto ML.NET
            var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
            // Data e ora
            var timestamp = DateTime.UtcNow;
            // Oggetto per la scrittura dei dati in memoria
            using var writer = new MemoryStream();
            ml.Data.SaveAsBinary(data, writer, true);
            BinaryData = writer.ToArray();
            Timestamp = timestamp;
         }
      }
      #endregion
   }
}

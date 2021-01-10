using Microsoft.ML;
using System;
using System.IO;

namespace MachineLearning
{
   /// <summary>
   /// Classe per lo storage di dati di tipo binario su file
   /// </summary>
   [Serializable]
   public sealed class DataStorageBinaryFile : DataStorageBinary, ITimestamp
   {
      #region Properties
      /// <summary>
      /// Path del file
      /// </summary>
      public string FilePath { get; private set; }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime Timestamp => File.GetLastWriteTimeUtc(FilePath);
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="filePath">Path del file</param>
      public DataStorageBinaryFile(string filePath)
      {
         if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException($"{nameof(filePath)} cannot be null or empty");
         FilePath = filePath;
      }
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream. Puo' essere null.
      /// </summary>
      /// <returns>Il path o null</returns>
      protected override string GetFilePath() => FilePath;
      /// <summary>
      /// Restituisce uno stream leggibile.
      /// </summary>
      /// <returns>Lo stream di lettura</returns>
      protected override Stream GetReadStream() => File.Exists(FilePath) ? File.OpenRead(FilePath) : new MemoryStream();
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      public override void SaveData(IMachineLearningContextProvider context, IDataView data)
      {
         using var stream = File.Create(FilePath);
         if (stream == null)
            throw new IOException($"Cannot write to file {FilePath}");
         SaveBinaryData(context, data, stream, KeepHidden);
      }
      #endregion
   }
}

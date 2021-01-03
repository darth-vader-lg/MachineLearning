using Microsoft.ML;
using System;
using System.IO;

namespace MachineLearning
{
   /// <summary>
   /// Classe per lo storage di dati di tipo file di testo
   /// </summary>
   [Serializable]
   public sealed class DataStorageTextFile : DataStorageText, ITimestamp
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
      public DataStorageTextFile(string filePath)
      {
         if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException($"{nameof(File)} cannot be null");
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
      protected override Stream GetReadStream() => File.OpenRead(FilePath);
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      public override IDataView LoadData(object context)
      {
         if (string.IsNullOrEmpty(FilePath) || !File.Exists(FilePath))
            throw new FileNotFoundException("File not found", FilePath);
         return base.LoadData(context);
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      public override void SaveData(object context, IDataView data)
      {
         using var stream = File.Create(FilePath);
         if (stream == null)
            throw new IOException($"Cannot write to file {FilePath}");
         SaveTextData(context, data, stream, SaveSchema, KeepHidden, ForceDense);
      }
      #endregion
   }
}

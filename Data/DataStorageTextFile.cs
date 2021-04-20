using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.IO;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo file di testo
   /// </summary>
   [Serializable]
   public sealed class DataStorageTextFile : DataStorageText, IDataTimestamp
   {
      #region Properties
      /// <summary>
      /// Path del file
      /// </summary>
      public string FilePath { get; private set; }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp => File.GetLastWriteTimeUtc(FilePath);
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="filePath">Path del file</param>
      public DataStorageTextFile(string filePath)
      {
         Contracts.CheckNonEmpty(filePath, nameof(filePath));
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
      /// <param name="textLoaderOptions">Opzioni di caricamento testuale</param>
      public override void SaveData(MLContext context, IDataAccess data, TextLoader.Options textLoaderOptions)
      {
         Contracts.CheckValue(context, nameof(context));
         using var stream = File.Create(FilePath);
         context.CheckIO(stream != null, $"Cannot write to file {FilePath}");
         SaveTextData(context, data, textLoaderOptions, stream, true);
      }
      #endregion
   }
}

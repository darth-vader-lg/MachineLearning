using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.IO;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo binario su file temporaneo
   /// </summary>
   [Serializable]
   public sealed class DataStorageBinaryTempFile : DataStorageBinary, IDataTimestamp
   {
      #region Fields
      /// <summary>
      /// Path del file
      /// </summary>
      [NonSerialized]
      private string _filePath;
      #endregion
      #region Properties
      /// <summary>
      /// Path del file
      /// </summary>
      public string FilePath => _filePath ??= Path.GetTempFileName();
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp => File.GetLastWriteTimeUtc(FilePath);
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public DataStorageBinaryTempFile() { }
      /// <summary>
      /// Finalizzatore
      /// </summary>
      ~DataStorageBinaryTempFile() => Dispose(false);
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Indicatore di dispose da programma</param>
      protected override void Dispose(bool disposing)
      {
         base.Dispose(disposing);
         if (_filePath != null) {
            try {
               FileUtil.Delete(_filePath);
               _filePath = null;
            }
            catch { }
         }
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
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="textLoaderOptions">Eventuali opzioni di caricamento testuale (non utilizzate per il salvataggio binario)</param>
      public override void SaveData(MLContext context, IDataAccess data, TextLoader.Options textLoaderOptions = default)
      {
         Contracts.CheckValue(context, nameof(context));
         using var stream = File.Create(FilePath);
         context.CheckIO(stream != null, $"Cannot write to file {FilePath}");
         SaveBinaryData(context, data, stream);
      }
      #endregion
   }
}

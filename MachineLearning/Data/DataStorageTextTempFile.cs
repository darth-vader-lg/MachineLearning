using MachineLearning.Util;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.IO;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo testo su file temporaneo
   /// </summary>
   [Serializable]
   public sealed class DataStorageTextTempFile : DataStorageText, IDataTimestamp, IDisposable
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
      /// Finalizzatore
      /// </summary>
      ~DataStorageTextTempFile() => Dispose(false);
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
      /// <param name="textLoaderOptions">Eventuali opzioni di caricamento testuale</param>
      public override void SaveData(IMachineLearningContextProvider context, IDataView data, TextLoader.Options textLoaderOptions = default)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         using var stream = File.Create(FilePath);
         context.ML.NET.CheckIO(stream != null, $"Cannot write to file {FilePath}");
         SaveTextData(context, data, textLoaderOptions, stream, true);
      }
      #endregion
   }
}

using MachineLearning.Util;
using Microsoft.ML;
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
      /// Costruttore
      /// </summary>
      public DataStorageTextTempFile() { }
      /// <summary>
      /// Finalizzatore
      /// </summary>
      ~DataStorageTextTempFile() => DeleteFile();
      /// <summary>
      /// Funzione di cancellazione del file
      /// </summary>
      private void DeleteFile()
      {
         if (_filePath != null) {
            try {
               FileUtil.Delete(_filePath);
               _filePath = null;
            }
            catch { }
         }
      }
      /// <summary>
      /// Dispose da codice
      /// </summary>
      public void Dispose()
      {
         DeleteFile();
         GC.SuppressFinalize(this);
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
      public override void SaveData(IMachineLearningContextProvider context, IDataView data)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         using var stream = File.Create(FilePath);
         context.ML.NET.CheckIO(stream != null, $"Cannot write to file {FilePath}");
         SaveTextData(context, data, stream, SaveSchema, KeepHidden, ForceDense);
      }
      #endregion
   }
}

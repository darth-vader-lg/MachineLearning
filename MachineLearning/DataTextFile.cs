using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

namespace MachineLearning
{
   /// <summary>
   /// Classe per lo storage di dati di tipo file di testo
   /// </summary>
   [Serializable]
   public sealed class DataTextFile : IDataStorage, IMultiStreamSource, ITimestamp
   {
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => 1;
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
      public DataTextFile(string filePath) => FilePath = filePath;
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Il path</returns>
      string IMultiStreamSource.GetPathOrNull(int index) => FilePath;
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      Stream IMultiStreamSource.Open(int index) => File.OpenRead(FilePath);
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      TextReader IMultiStreamSource.OpenTextReader(int index) => new StreamReader(FilePath);
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(object context)
      {
         if (string.IsNullOrEmpty(FilePath) || !File.Exists(FilePath))
            throw new FileNotFoundException("File not found", FilePath);
         var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
         var opt = (context as ITextLoaderOptionsProvider)?.TextLoaderOptions ?? new TextLoader.Options();
         return ml.Data.CreateTextLoader(opt).Load(this);
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
            // Oggetto per la scrittura dei dati in memoria
            var tmpPath = default(string);
            var writer = default(FileStream);
            try {
               tmpPath = Path.GetTempFileName();
               writer = File.Create(tmpPath);
               // Contesto e opzioni
               var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
               var opt = (context as ITextLoaderOptionsProvider)?.TextLoaderOptions ?? new TextLoader.Options();
               // Separatore di colonne
               var separator = opt.Separators?.FirstOrDefault() ?? '\t';
               separator = separator != default ? separator : '\t';
               // Salva come testo i dati
               ml.Data.SaveAsText(
                  data: data,
                  stream: writer,
                  separatorChar: separator,
                  headerRow: opt.HasHeader,
                  schema: schema,
                  keepHidden: true,
                  forceDense: false);
               writer.Close();
               writer = default;
               File.Copy(tmpPath, FilePath, true);
            }
            finally {
               try {
                  if (writer != default)
                     writer.Close();
                  if (tmpPath != default && File.Exists(tmpPath))
                     File.Delete(tmpPath);
               }
               catch (Exception) { }
            }
         }
      }
      #endregion
   }
}

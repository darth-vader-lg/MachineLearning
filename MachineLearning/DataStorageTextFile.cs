using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MachineLearning
{
   /// <summary>
   /// Classe per lo storage di dati di tipo file di testo
   /// </summary>
   [Serializable]
   public sealed partial class DataStorageTextFile : IDataStorage, IMultiStreamSource, ITimestamp
   {
      #region Fields
      /// <summary>
      /// Sorgente per il caricamento
      /// </summary>
      [NonSerialized]
      private Source _source;
      #endregion
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => (_source ??= new Source(this)).Count;
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
      public DataStorageTextFile(string filePath) => FilePath = filePath;
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Il path</returns>

      string IMultiStreamSource.GetPathOrNull(int index) => (_source ??= new Source(this)).GetPathOrNull(index);
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      Stream IMultiStreamSource.Open(int index) => (_source ??= new Source(this)).Open(index);
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      TextReader IMultiStreamSource.OpenTextReader(int index) => (_source ??= new Source(this)).OpenTextReader(index);
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="opt">Opzioni di testo</param>
      /// <param name="extra">Sorgenti extra di dati</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(MachineLearningContext ml, TextLoaderOptions opt = default, params IMultiStreamSource[] extra)
      {
         if (string.IsNullOrEmpty(FilePath) || !File.Exists(FilePath))
            throw new FileNotFoundException("File not found", FilePath);
         return ml.NET.Data.CreateTextLoader(opt ?? new TextLoaderOptions()).Load(_source = new Source(this, extra));
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="opt">Opzioni di testo</param>
      /// <param name="schema">Commento contenente lo schema nei dati di tipo file testuali (ignorato negli altri)</param>
      /// <param name="extra">Eventuali altri stream di dati</param>
      public void SaveData(MachineLearningContext ml, IDataView data, TextLoaderOptions opt = default, bool schema = false, params IMultiStreamSource[] extra)
      {
         lock (this) {
            // Oggetto per la scrittura dei dati in memoria
            var tmpPath = default(string);
            var writer = default(FileStream);
            try {
               tmpPath = Path.GetTempFileName();
               writer = File.Create(tmpPath);
               // Opzioni
               opt ??= new TextLoaderOptions();
               // Separatore di colonne
               var separator = opt.Separators?.FirstOrDefault() ?? '\t';
               separator = separator != default ? separator : '\t';
               // Salva come testo i dati
               ml.NET.Data.SaveAsText(
                  data: data,
                  stream: writer,
                  separatorChar: separator,
                  headerRow: opt.HasHeader,
                  schema: schema,
                  keepHidden: false,
                  forceDense: false);
               // Salva gli stream extra
               foreach (var item in extra) {
                  var loader = ml.NET.Data.CreateTextLoader(opt);
                  data = loader.Load(item);
                  ml.NET.Data.SaveAsText(
                     data: data,
                     stream: writer,
                     separatorChar: separator,
                     headerRow: opt.HasHeader,
                     schema: schema,
                     keepHidden: false,
                     forceDense: false);
               }
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

   /// <summary>
   /// La sorgente dei dati
   /// </summary>
   public partial class DataStorageTextFile
   {
      private class Source : IMultiStreamSource
      {
         #region Fields
         /// <summary>
         /// Sorgenti ed indici
         /// </summary>
         private readonly (IMultiStreamSource Source, int Index)[] _total;
         /// <summary>
         /// Testo
         /// </summary>
         private readonly DataStorageTextFile _owner;
         #endregion
         #region Properties
         /// <summary>
         /// Il numero di items
         /// </summary>
         public int Count => _total.Length;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenzqa</param>
         /// <param name="extra">Sorgenti extra di dati</param>
         public Source(DataStorageTextFile owner, params IMultiStreamSource[] extra)
         {
            this._owner = owner;
            var indices = new List<(IMultiStreamSource Source, int Index)>
            {
               (Source: this, Index: 0)
            };
            foreach (var e in extra) {
               foreach (var ix in Enumerable.Range(0, e.Count))
                  indices.Add((Source: e, Index: ix));
            }
            _total = indices.ToArray();
         }
         /// <summary>
         /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Sempre null</returns>
         public string GetPathOrNull(int index) => index == 0 ? _owner.FilePath : _total[index].Source.GetPathOrNull(_total[index].Index);
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public Stream Open(int index) => index == 0 ? File.OpenRead(_owner.FilePath) : _total[index].Source.Open(_total[index].Index);
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public TextReader OpenTextReader(int index) => index == 0 ? new StreamReader(_owner.FilePath) : _total[index].Source.OpenTextReader(_total[index].Index);
         #endregion
      }
   }
}

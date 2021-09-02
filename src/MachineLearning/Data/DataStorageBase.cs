using Microsoft.ML.Data;
using System;
using System.IO;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe base per lo storage di dati
   /// </summary>
   [Serializable]
   public abstract class DataStorageBase : IDisposable, IMultiStreamSource
   {
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => 1;
      /// <summary>
      /// Indicatore di oggetto disposed
      /// </summary>
      public bool IsDisposed { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Dispose da programma
      /// </summary>
      public void Dispose()
      {
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Indicatore di dispose da programma</param>
      protected virtual void Dispose(bool disposing) => IsDisposed = true;
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream. Puo' essere null.
      /// </summary>
      /// <returns>Il path o null</returns>
      protected virtual string GetFilePath() => null;
      /// <summary>
      /// Restituisce uno stream leggibile.
      /// </summary>
      /// <returns>Lo stream di lettura</returns>
      protected virtual Stream GetReadStream() => null;
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Il path</returns>
      string IMultiStreamSource.GetPathOrNull(int index) => GetFilePath();
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      Stream IMultiStreamSource.Open(int index) => GetReadStream();
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      TextReader IMultiStreamSource.OpenTextReader(int index) => new StreamReader(GetReadStream());
      #endregion
   }
}

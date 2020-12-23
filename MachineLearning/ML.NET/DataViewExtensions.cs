using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML
{
   /// <summary>
   /// Estensioni per l'accesso ai gestori di dati
   /// </summary>
   public static class DataViewExtensions
   {
      #region Methods
      /// <summary>
      /// Restituisce il valore float di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="column">Indice colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static float GetFloat(this IDataView dataView, int column, int row = 0) => dataView.GetValue<float>(column, row);
      /// <summary>
      /// Restituisce il valore float di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="columnName">Nome colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static float GetFloat(this IDataView dataView, string columnName, int row = 0) => dataView.GetValue<float>(columnName, row);
      /// <summary>
      /// Restituisce il valore stringa di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="column">Indice colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static string GetString(this IDataView dataView, int column, int row = 0) => dataView.GetValue<ReadOnlyMemory<char>>(column, row).ToString();
      /// <summary>
      /// Restituisce il valore stringa di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="columnName">Nome colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static string GetString(this IDataView dataView, string columnName, int row = 0) => dataView.GetValue<ReadOnlyMemory<char>>(columnName, row).ToString();
      /// <summary>
      /// Restituisce il valore di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="column">Indice colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static T GetValue<T>(this IDataView dataView, int column, int row = 0)
      {
         var cursor = dataView.GetRowCursor(new[] { dataView.Schema[column] });
         while (cursor.MoveNext()) {
            if (cursor.Position == row) {
               var val = default(T);
               cursor.GetGetter<T>(dataView.Schema[column]).Invoke(ref val);
               return val;
            }
         }
         return default;
      }
      /// <summary>
      /// Restituisce il valore di una cella
      /// </summary>
      /// <param name="dataView">Dati</param>
      /// <param name="columnName">Nome colonna</param>
      /// <param name="row">Indice riga</param>
      /// <returns>Il valore</returns>
      public static T GetValue<T>(this IDataView dataView, string columnName, int row = 0)
      {
         var col = dataView.Schema.LastOrDefault(c => !c.IsHidden && c.Name == columnName);
         if (col.Name != columnName)
            return default;
         return dataView.GetValue<T>(col.Index, row);
      }
      #endregion
   }
}

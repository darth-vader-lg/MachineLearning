using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning.Data
{
   /// <summary>
   /// Estensioni per l'accesso ai gestori di dati
   /// </summary>
   public static class DataViewExtensions
   {
      #region Methods
      /// <summary>
      /// Trasforma un cursore in enumerabile
      /// </summary>
      /// <param name="cursor">Cursore di dato</param>
      /// <returns>L'enumerabile</returns>
      public static IEnumerable<DataViewRowCursor> AsEnumerable(this DataViewRowCursor cursor)
      {
         while (cursor.MoveNext())
            yield return cursor;
      }
      /// <summary>
      /// Restituisce un valore
      /// </summary>
      /// <param name="cursor">Il cursore</param>
      /// <param name="column">La colonna</param>
      /// <returns>Il valore</returns>
      public static DataViewValue GetValue(this DataViewRowCursor cursor, DataViewSchema.Column column) => cursor.GetValue(column.Index);
      /// <summary>
      /// Restituisce un valore
      /// </summary>
      /// <param name="cursor">Il cursore</param>
      /// <param name="columnIndex">L'indice di colonna</param>
      /// <returns>Il valore</returns>
      public static DataViewValue GetValue(this DataViewRowCursor cursor, int columnIndex)
      {
         // Crea il metodo generico del getter
         var getterMethodInfo = typeof(DataViewExtensions).GetMethod(nameof(GetValue), 1, new[] { typeof(DataViewRowCursor), typeof(int) });
         var getterGenericMethodInfo = getterMethodInfo.MakeGenericMethod(cursor.Schema[columnIndex].Type.RawType);
         // Lo invoca e restituisce il risultato
         return new DataViewValue(getterGenericMethodInfo.Invoke(null, new object[] { cursor, columnIndex }));
      }
      /// <summary>
      /// Restituisce un valore
      /// </summary>
      /// <param name="cursor">Il cursore</param>
      /// <param name="columnName">Il nome della colonna</param>
      /// <returns>Il valore</returns>
      public static DataViewValue GetValue(this DataViewRowCursor cursor, string columnName) => cursor.GetValue(cursor.Schema[columnName]);
      /// <summary>
      /// Restituisce un valore
      /// </summary>
      /// <typeparam name="T">Tipo di valore</typeparam>
      /// <param name="cursor">Il cursore</param>
      /// <param name="col">L'indice di colonna</param>
      /// <returns>Il valore</returns>
      public static T GetValue<T>(this DataViewRowCursor cursor, int col)
      {
         // Azione di restituzione dei valori
         var getter = cursor.GetGetter<T>(cursor.Schema[col]);
         T value = default;
         getter(ref value);
         return value;
      }
      /// <summary>
      /// Crea una IDataAccess che e' il merge dell'oggetto principale con una serie di altre IDataView
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="others">Altre viste di dati da concatenare</param>
      /// <returns>La vista di dati concatenata</returns>
      public static IDataAccess Merge(this IDataAccess data, params IDataAccess[] others) =>
         DataViewMerged.Create(data.ML, data.Schema, new[] { data }.Concat(others).ToArray());
      /// <summary>
      /// Equivalent to calling Equals(ColumnType) for non-vector types. For vector type,
      /// returns true if current and other vector types have the same size and item type.
      /// </summary>
      public static bool SameSizeAndItemType(this DataViewType columnType, DataViewType other)
      {
         if (other == null)
            return false;
         if (columnType.Equals(other))
            return true;
         // For vector types, we don't care about the factoring of the dimensions.
         if (columnType is not VectorDataViewType vectorType || other is not VectorDataViewType otherVectorType)
            return false;
         if (!vectorType.ItemType.Equals(otherVectorType.ItemType))
            return false;
         return vectorType.Size == otherVectorType.Size;
      }
      /// <summary>
      /// Trasforma la IDataAccess in una data view filtrata
      /// </summary>
      /// <param name="data">Dati</param>
      /// <param name="filter">Filtro</param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewFiltered ToDataViewFiltered(this IDataAccess data, DataViewRowFilter filter) => DataViewFiltered.Create(data, filter);
      /// <summary>
      /// Trasforma la IDataAccess in una griglia di dati
      /// </summary>
      /// <param name="data">Dati</param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewGrid ToDataViewGrid(this IDataAccess data) => DataViewGrid.Create(data);
      /// <summary>
      /// Trasforma un cursore di vista dati in una riga con valori
      /// </summary>
      /// <param name="cursor">Cursore</param>
      /// <param name="context">Contesto</param>
      /// <returns>La riga di dati</returns>
      public static DataViewValuesRow ToDataViewValuesRow(this DataViewRowCursor cursor, IMachineLearningContext context) => DataViewValuesRow.Create(context, cursor);
      #endregion
   }
}
